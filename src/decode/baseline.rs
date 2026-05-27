//! Baseline Huffman scan decoder (SOF0).
//!
//! Orchestrates one all-components-interleaved scan: for each MCU
//! in raster order, decode the per-component `Hi * Vi` 8x8 blocks
//! (DC diff → AC RLE+magnitude → de-zig-zag → dequantize → IDCT).
//! The output is a set of per-component sample planes ready for
//! chroma upsample + YCbCr→RGB.

use crate::arch;
use crate::tables::ZIGZAG;

use super::error::{DecodeError, Result};
use super::huffman::{
    BitReader, FastAcHuffmanTable, FastDcHuffmanTable, HuffmanDecodeTable, decode_ac_fast,
    decode_dc_fast, extend,
};
use super::markers::{Component, DecoderHeaders, QuantTable, ScanHeader};

/// AC Huffman decoder bundle: the canonical slow-path table plus the
/// 10-bit combined LUT that fuses `decode_symbol` + `get_bits` into a
/// single peek/drop on a hit. The scan loop tries `decode_ac_fast`
/// against `fast` first and falls back to `decode_symbol(&slow)` +
/// `get_bits` when the LUT slot is invalid (long codes or codes whose
/// magnitude bits spill past the peek window).
pub struct AcTablePair {
    pub slow: HuffmanDecodeTable,
    pub fast: FastAcHuffmanTable,
}

/// DC Huffman decoder bundle, mirroring [`AcTablePair`]: the canonical
/// slow-path table plus the 10-bit combined LUT that fuses
/// `decode_symbol` + `get_bits` for the single per-block DC term. The
/// per-block DC contribution to scan-loop cycles is small (one term vs.
/// up to 63 AC terms), so the bundle exists mainly to extend the fast
/// path's coverage to every Huffman lookup in the baseline scan.
pub struct DcTablePair {
    pub slow: HuffmanDecodeTable,
    pub fast: FastDcHuffmanTable,
}

/// Decoded image data: per-component pixel planes (post-IDCT, post-
/// level-shift). Each plane is `comp.plane_width × comp.plane_height`
/// u8 samples, row-major, with the per-row stride equal to plane_width
/// (no padding beyond what's needed to align to 8-pixel block edges).
pub struct DecodedPlanes {
    pub width: u32,
    pub height: u32,
    pub components: Vec<DecodedPlane>,
}

#[allow(dead_code)] // padded_height kept for diagnostic / future fancy-upsample
pub struct DecodedPlane {
    /// Frame component descriptor (sampling factors + qt index).
    pub component: Component,
    /// Block-aligned plane width (multiple of 8).
    pub stride: usize,
    /// Block-aligned plane height (multiple of 8).
    pub padded_height: usize,
    /// True plane width (`ceil(image_width * Hi / Hmax)`).
    pub plane_width: usize,
    /// True plane height (`ceil(image_height * Vi / Vmax)`).
    pub plane_height: usize,
    /// Sample buffer (`stride * padded_height` bytes, row-major).
    pub samples: Vec<u8>,
}

/// Allocate per-component output planes sized for the frame's MCU
/// grid. The buffers are block-aligned (multiples of 8 in each
/// direction) so scan-time block writes don't have to special-case
/// the right / bottom image edge; the `plane_width` / `plane_height`
/// fields record the true dimensions for the consumer to trim by.
pub fn allocate_planes(frame: &super::markers::FrameHeader) -> Vec<DecodedPlane> {
    let h_max = frame.h_max();
    let v_max = frame.v_max();
    let mcus_x = (frame.width as u32).div_ceil((h_max as u32) * 8);
    let mcus_y = (frame.height as u32).div_ceil((v_max as u32) * 8);
    let mut planes = Vec::with_capacity(frame.components.len());
    for comp in &frame.components {
        let pw = (frame.width as u32 * comp.h as u32).div_ceil(h_max as u32) as usize;
        let ph = (frame.height as u32 * comp.v as u32).div_ceil(v_max as u32) as usize;
        let stride = (mcus_x as usize) * (comp.h as usize) * 8;
        let padded_height = (mcus_y as usize) * (comp.v as usize) * 8;
        // Skip the per-decode zero-fill: every byte at indices
        // `[0, stride * padded_height)` is overwritten by exactly one
        // `place_block` write during the scan (the buffer is block-
        // aligned, so the right / bottom edge blocks fully cover the
        // padded area). For 4K 4:2:0 this saves ~12 MB of zero-fill
        // page-fault cost per decode.
        // Safety: `u8` has no validity invariants, so `set_len` on a
        // freshly allocated Vec<u8> without initialization is sound;
        // the "fully written before read" contract above keeps it
        // from being read-as-uninit (= no UB).
        let mut samples: Vec<u8> = Vec::with_capacity(stride * padded_height);
        #[allow(clippy::uninit_vec)]
        unsafe {
            samples.set_len(stride * padded_height);
        }
        planes.push(DecodedPlane {
            component: *comp,
            stride,
            padded_height,
            plane_width: pw,
            plane_height: ph,
            samples,
        });
    }
    planes
}

/// Run a single baseline Huffman scan, writing decoded pixels into the
/// pre-allocated `planes`. Returns the byte position immediately past
/// the marker that terminated entropy data, together with that
/// marker's code (see [`super::progressive`] for the rationale on
/// returning both).
///
/// Two iteration modes:
/// - **Interleaved** (`scan.components.len() > 1`): MCU-major raster
///   walk, decoding `Hi × Vi` blocks per component inside each MCU.
///   This is the common case for "single SOS contains all components".
/// - **Non-interleaved** (`scan.components.len() == 1`): walk the one
///   scanned component's own block grid in raster order. Baseline
///   JPEGs split across multiple per-component SOS markers route here;
///   each scan fills its component's plane.
#[allow(clippy::too_many_arguments)]
pub fn decode_baseline_scan_into(
    src: &[u8],
    entropy_start: usize,
    headers: &DecoderHeaders,
    scan: &ScanHeader,
    dc_tables: &[Option<DcTablePair>; 4],
    ac_tables: &[Option<AcTablePair>; 4],
    qt_by_id: &[Option<&QuantTable>; 4],
    planes: &mut [DecodedPlane],
) -> Result<(usize, Option<u8>)> {
    let frame = &headers.frame;
    let h_max = frame.h_max();
    let v_max = frame.v_max();
    let mcu_w_pixels = (h_max as u32) * 8;
    let mcu_h_pixels = (v_max as u32) * 8;
    let mcus_x = (frame.width as u32).div_ceil(mcu_w_pixels);
    let mcus_y = (frame.height as u32).div_ceil(mcu_h_pixels);

    let mut br = BitReader::new(src, entropy_start);
    let mut prev_dc = [0i32; 4];
    let restart_interval = headers.restart_interval as u32;
    let mut mcus_since_restart: u32 = 0;
    let mut expected_rst: u8 = 0;
    let mut nat_coef = [0i16; 64];

    if scan.components.len() > 1 {
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                if restart_interval > 0 && mcus_since_restart == restart_interval {
                    handle_restart(&mut br, &mut prev_dc, &mut expected_rst)?;
                    mcus_since_restart = 0;
                }
                for (scan_idx, sc) in scan.components.iter().enumerate() {
                    let (comp_idx, comp) = find_component(frame, sc.component_id)?;
                    let plane = &mut planes[comp_idx];
                    let dc_tbl = dc_tables[sc.dc_table as usize]
                        .as_ref()
                        .ok_or(DecodeError::Malformed("scan refers to undefined DC table"))?;
                    let ac_tbl = ac_tables[sc.ac_table as usize]
                        .as_ref()
                        .ok_or(DecodeError::Malformed("scan refers to undefined AC table"))?;
                    let qt = qt_by_id[comp.qt as usize].ok_or(DecodeError::Malformed(
                        "scan refers to undefined quant table",
                    ))?;
                    for v_block in 0..(comp.v as u32) {
                        for h_block in 0..(comp.h as u32) {
                            nat_coef.fill(0);
                            decode_block_baseline(
                                &mut br,
                                dc_tbl,
                                ac_tbl,
                                qt,
                                &mut prev_dc[scan_idx],
                                &mut nat_coef,
                            )?;
                            let mut block = [0u8; 64];
                            arch::backend::dct::idct_islow(&nat_coef, &mut block);
                            let base_x = (mx * comp.h as u32 + h_block) as usize * 8;
                            let base_y = (my * comp.v as u32 + v_block) as usize * 8;
                            place_block(plane, base_x, base_y, &block);
                        }
                    }
                }
                mcus_since_restart += 1;
            }
        }
    } else {
        // Non-interleaved scan: per T.81 A.2.2/A.2.4 the MCU collapses
        // to a single data unit and the component's blocks are walked
        // in left-to-right top-to-bottom raster order. Block counts
        // come from the spec formula `ceil(W*Hi/(Hmax*8))`, NOT from
        // `mcus_x * Hi` — for unaligned widths (e.g. 65x65 4:2:2) the
        // two diverge and `mcus_x * Hi` over-counts by one block per
        // row, de-syncing the bit reader the same way the progressive
        // path tripped on synthetic_image.jpg.
        let sc = &scan.components[0];
        let (comp_idx, comp) = find_component(frame, sc.component_id)?;
        let dc_tbl = dc_tables[sc.dc_table as usize]
            .as_ref()
            .ok_or(DecodeError::Malformed("scan refers to undefined DC table"))?;
        let ac_tbl = ac_tables[sc.ac_table as usize]
            .as_ref()
            .ok_or(DecodeError::Malformed("scan refers to undefined AC table"))?;
        let qt = qt_by_id[comp.qt as usize].ok_or(DecodeError::Malformed(
            "scan refers to undefined quant table",
        ))?;
        let plane = &mut planes[comp_idx];
        let blocks_x = (frame.width as usize * comp.h as usize)
            .div_ceil(h_max as usize * 8)
            .max(1);
        let blocks_y = (frame.height as usize * comp.v as usize)
            .div_ceil(v_max as usize * 8)
            .max(1);
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                if restart_interval > 0 && mcus_since_restart == restart_interval {
                    handle_restart(&mut br, &mut prev_dc, &mut expected_rst)?;
                    mcus_since_restart = 0;
                }
                nat_coef.fill(0);
                decode_block_baseline(
                    &mut br,
                    dc_tbl,
                    ac_tbl,
                    qt,
                    &mut prev_dc[0],
                    &mut nat_coef,
                )?;
                let mut block = [0u8; 64];
                arch::backend::dct::idct_islow(&nat_coef, &mut block);
                place_block(plane, bx * 8, by * 8, &block);
                mcus_since_restart += 1;
            }
        }
    }

    Ok((br.pos(), br.marker()))
}

/// Decode a (possibly multi-scan) baseline JPEG by looping over SOS
/// markers, threading any intervening DHT / DQT / DRI segments back
/// into the working header tables, and dispatching each scan into the
/// shared `planes` buffer via [`decode_baseline_scan_into`]. Handles
/// both the common single-SOS case and the rarer
/// multi-SOS-per-component split that some encoders emit.
pub fn decode_baseline_multi(
    src: &[u8],
    entropy_start: usize,
    headers: &DecoderHeaders,
    first_scan: ScanHeader,
) -> Result<DecodedPlanes> {
    let frame = &headers.frame;
    let mut planes = allocate_planes(frame);
    let mut huffman_specs = headers.huffman.clone();
    let mut quant_tables = headers.quant.clone();
    let mut restart_interval = headers.restart_interval;

    let mut scan: Option<ScanHeader> = Some(first_scan);
    let mut pos = entropy_start;
    let mut headers_working = headers.clone();
    while let Some(s) = scan {
        let (dc_tables, ac_tables) = build_huffman_tables(&huffman_specs)?;
        let qt_by_id = index_quant_tables(&quant_tables);
        headers_working.restart_interval = restart_interval;
        let (after_pos, pending_marker) = decode_baseline_scan_into(
            src,
            pos,
            &headers_working,
            &s,
            &dc_tables,
            &ac_tables,
            &qt_by_id,
            &mut planes,
        )?;
        pos = after_pos;

        let mut mr = super::markers::MarkerReader::resume_at(src, pos);
        scan = mr.next_scan_or_end(
            frame,
            &mut huffman_specs,
            &mut quant_tables,
            &mut restart_interval,
            pending_marker,
        )?;
        pos = mr.pos();
    }

    Ok(DecodedPlanes {
        width: frame.width as u32,
        height: frame.height as u32,
        components: planes,
    })
}

/// Decode the next 8x8 block and write the dequantized natural-order
/// coefficients into `nat_coef`. `prev_dc` is updated on exit.
///
/// Caller must zero `nat_coef` before each call — only non-zero
/// positions are written here (the entropy stream's RLE structure
/// skips runs of zero coefficients, and writing them again would just
/// re-zero something we already pre-zeroed).
///
/// Fusing dequant into entropy decode eliminates the intermediate
/// `zz_coef` buffer and the separate `for k in 0..64` dequant loop:
/// each AC magnitude flows directly into `nat_coef[ZIGZAG[k]]`
/// pre-multiplied by `qt.values[ZIGZAG[k]]`. On natural-content the
/// AC scan terminates early (EOB), so the dequant loop's 64-iter
/// constant cost disappears entirely on sparse blocks.
fn decode_block_baseline(
    br: &mut BitReader,
    dc_tbl: &DcTablePair,
    ac_tbl: &AcTablePair,
    qt: &QuantTable,
    prev_dc: &mut i32,
    nat_coef: &mut [i16; 64],
) -> Result<()> {
    // ---- DC term ----
    // Try the combined DC LUT first; on a miss the bit buffer is
    // untouched and we fall back to the canonical decode_symbol +
    // get_bits path. The fast path collapses both peek/drop pairs
    // into one whenever code_length + size fits the peek window.
    let (dc_size, dc_bits) = match decode_dc_fast(br, &dc_tbl.fast)? {
        Some(fast) => (fast.size as u32, fast.magnitude_raw),
        None => {
            let t = br.decode_symbol(&dc_tbl.slow)?;
            let size = t as u32;
            let bits = if size > 0 { br.get_bits(size)? } else { 0 };
            (size, bits)
        }
    };
    let dc_diff = if dc_size == 0 {
        0
    } else {
        extend(dc_bits, dc_size)
    };
    *prev_dc = prev_dc.wrapping_add(dc_diff);
    // ZIGZAG[0] = 0; DC always lands at natural position 0.
    nat_coef[0] = (*prev_dc as i16).wrapping_mul(qt.values[0] as i16);

    // ---- AC terms ----
    // Try the combined LUT first; on a miss the bit buffer is
    // untouched and we fall back to the canonical decode_symbol +
    // get_bits path. The fast path collapses peek/drop pairs from
    // two to one when both the code and its magnitude bits fit in
    // FAST_AC_PEEK_WIDTH bits — true for the bulk of standard-table
    // baseline AC symbols (~95% slot coverage).
    let mut k: usize = 1;
    while k < 64 {
        let (run, size, mag_raw) = match decode_ac_fast(br, &ac_tbl.fast)? {
            Some(fast) => (fast.run as usize, fast.size as u32, fast.magnitude_raw),
            None => {
                let rs = br.decode_symbol(&ac_tbl.slow)?;
                let run = (rs >> 4) as usize;
                let size = (rs & 0x0F) as u32;
                let mag_raw = if size > 0 { br.get_bits(size)? } else { 0 };
                (run, size, mag_raw)
            }
        };
        if size == 0 {
            if run == 15 {
                // ZRL: 16 zeros, no value.
                k += 16;
                continue;
            }
            // EOB (run == 0 && size == 0): rest of block is zero.
            break;
        }
        k += run;
        if k >= 64 {
            return Err(DecodeError::Malformed("AC run exceeded block"));
        }
        let mag = extend(mag_raw, size) as i16;
        let nat_idx = ZIGZAG[k];
        nat_coef[nat_idx] = mag.wrapping_mul(qt.values[nat_idx] as i16);
        k += 1;
    }
    Ok(())
}

fn handle_restart(br: &mut BitReader, prev_dc: &mut [i32; 4], expected: &mut u8) -> Result<()> {
    // Drop pending bits — RSTn realigns the entropy stream on a byte
    // boundary (T.81 F.1.5).
    br.reset_bit_buffer();
    // Pull bytes until the reader surfaces a marker. `get_bits(8)`
    // returns `Err(UnexpectedEof)` if the stream is truncated before a
    // marker shows up; we **must** propagate that — silently swallowing
    // the error would let a crafted JPEG (truncated mid-restart) spin
    // this loop forever (CVE-class DoS).
    while br.marker().is_none() {
        br.get_bits(8)?;
    }
    let m = br
        .marker()
        .expect("marker is Some — while loop only exits on Some");
    if !(0xD0..=0xD7).contains(&m) {
        return Err(DecodeError::Malformed("expected RSTn between intervals"));
    }
    // libjpeg-turbo is lenient about RST sequence — we accept any
    // `rst_n` rather than failing on `rst_n != *expected & 7`, but the
    // counter is still advanced so future audit / debug output stays
    // honest about the expected slot.
    *expected = (*expected + 1) & 7;
    br.clear_marker();
    *prev_dc = [0; 4];
    Ok(())
}

fn find_component(frame: &super::markers::FrameHeader, id: u8) -> Result<(usize, Component)> {
    frame
        .components
        .iter()
        .enumerate()
        .find(|(_, c)| c.id == id)
        .map(|(i, c)| (i, *c))
        .ok_or(DecodeError::Malformed("scan refers to unknown component"))
}

fn place_block(plane: &mut DecodedPlane, base_x: usize, base_y: usize, block: &[u8; 64]) {
    let stride = plane.stride;
    for j in 0..8 {
        let dst_off = (base_y + j) * stride + base_x;
        plane.samples[dst_off..dst_off + 8].copy_from_slice(&block[j * 8..j * 8 + 8]);
    }
}

/// One DC decoder bundle per Huffman destination id (0..=3).
pub type DcTableSlots = [Option<DcTablePair>; 4];
/// One AC decoder bundle per Huffman destination id (0..=3).
pub type AcTableSlots = [Option<AcTablePair>; 4];

/// Scan the supplied DHT specs into the `(dc, ac)` decode-table arrays
/// (indexed by destination id 0..=3). Both DC and AC slots carry the
/// canonical slow-path table and the combined fast-path LUT so the
/// scan loop can take whichever path each symbol fits.
pub fn build_huffman_tables(
    specs: &[super::markers::HuffmanTableSpec],
) -> Result<(DcTableSlots, AcTableSlots)> {
    let mut dc: DcTableSlots = Default::default();
    let mut ac: AcTableSlots = Default::default();
    for spec in specs {
        match spec.class {
            0 => {
                let slow = HuffmanDecodeTable::from_spec(spec)?;
                let fast = FastDcHuffmanTable::from_spec(spec)?;
                dc[spec.id as usize] = Some(DcTablePair { slow, fast });
            }
            1 => {
                let slow = HuffmanDecodeTable::from_spec(spec)?;
                let fast = FastAcHuffmanTable::from_spec(spec)?;
                ac[spec.id as usize] = Some(AcTablePair { slow, fast });
            }
            _ => return Err(DecodeError::Malformed("DHT class > 1")),
        }
    }
    Ok((dc, ac))
}

/// Lay out the quant tables into an `id → &QuantTable` array.
pub fn index_quant_tables<'a>(qts: &'a [QuantTable]) -> [Option<&'a QuantTable>; 4] {
    let mut out: [Option<&'a QuantTable>; 4] = [None; 4];
    for qt in qts {
        out[qt.id as usize] = Some(qt);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: previously, `handle_restart` swallowed `Err` from
    /// `get_bits(8)` with `let _ = …`, which let a truncated JPEG (no
    /// RST marker arrives) spin the loop forever — a cheap DoS via
    /// crafted input. The fix propagates the error; this test pins it
    /// down by handing the function an empty entropy stream and
    /// asserting it returns within finite time with `UnexpectedEof`.
    #[test]
    fn handle_restart_errors_on_truncated_input() {
        let bytes: &[u8] = &[];
        let mut br = BitReader::new(bytes, 0);
        let mut prev_dc = [42i32; 4];
        let mut expected = 0u8;
        let err = handle_restart(&mut br, &mut prev_dc, &mut expected).unwrap_err();
        assert!(
            matches!(err, DecodeError::UnexpectedEof),
            "expected UnexpectedEof, got {err:?}",
        );
        // The function bailed before reaching the predictor reset.
        assert_eq!(prev_dc, [42, 42, 42, 42]);
    }

    #[test]
    fn handle_restart_consumes_rst_marker_and_resets_predictors() {
        // RST3 marker (0xFF 0xD3) with no preceding entropy bytes.
        let bytes: &[u8] = &[0xFF, 0xD3];
        let mut br = BitReader::new(bytes, 0);
        let mut prev_dc = [123i32; 4];
        let mut expected = 3u8;
        handle_restart(&mut br, &mut prev_dc, &mut expected).expect("RST3 should be accepted");
        assert_eq!(prev_dc, [0; 4]);
        assert_eq!(expected, 4);
    }

    #[test]
    fn handle_restart_rejects_non_rst_marker() {
        // 0xFF 0xD9 = EOI, not an RST.
        let bytes: &[u8] = &[0xFF, 0xD9];
        let mut br = BitReader::new(bytes, 0);
        let mut prev_dc = [0i32; 4];
        let mut expected = 0u8;
        let err = handle_restart(&mut br, &mut prev_dc, &mut expected).unwrap_err();
        assert!(matches!(err, DecodeError::Malformed(_)));
    }
}
