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
use super::huffman::{BitReader, HuffmanDecodeTable, extend};
use super::markers::{Component, DecoderHeaders, QuantTable, ScanHeader};

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

/// Run a baseline Huffman scan, decoding all interleaved components
/// in raster MCU order. `entropy_start` is the byte offset where the
/// entropy-coded data begins (immediately after SOS).
///
/// Returns the `DecodedPlanes` and the byte position immediately past
/// the entropy data (= at the byte after the terminating marker, with
/// the marker code reachable via the bit-reader's `marker()`).
#[allow(clippy::too_many_arguments)]
pub fn decode_baseline_scan<'a>(
    src: &'a [u8],
    entropy_start: usize,
    headers: &DecoderHeaders,
    scan: &ScanHeader,
    dc_tables: &[Option<HuffmanDecodeTable>; 4],
    ac_tables: &[Option<HuffmanDecodeTable>; 4],
    qt_by_id: &[Option<&QuantTable>; 4],
) -> Result<(DecodedPlanes, BitReader<'a>)> {
    let frame = &headers.frame;
    let h_max = frame.h_max();
    let v_max = frame.v_max();
    let mcu_w_pixels = (h_max as u32) * 8;
    let mcu_h_pixels = (v_max as u32) * 8;
    let mcus_x = (frame.width as u32).div_ceil(mcu_w_pixels);
    let mcus_y = (frame.height as u32).div_ceil(mcu_h_pixels);

    // Allocate per-component planes, sized to a multiple of 8 for
    // block-aligned writes; we trim back to the true dimensions when
    // returning.
    let mut planes: Vec<DecodedPlane> = Vec::with_capacity(frame.components.len());
    for comp in &frame.components {
        // True plane dims (before block-alignment padding).
        let pw = (frame.width as u32 * comp.h as u32).div_ceil(h_max as u32) as usize;
        let ph = (frame.height as u32 * comp.v as u32).div_ceil(v_max as u32) as usize;
        // Block-aligned dims (used for the actual sample buffer so the
        // per-block writes don't have to special-case the right/bottom
        // image edge).
        let stride = (mcus_x as usize) * (comp.h as usize) * 8;
        let padded_height = (mcus_y as usize) * (comp.v as usize) * 8;
        let samples = vec![0u8; stride * padded_height];
        planes.push(DecodedPlane {
            component: *comp,
            stride,
            padded_height,
            plane_width: pw,
            plane_height: ph,
            samples,
        });
    }

    let mut br = BitReader::new(src, entropy_start);
    // DC predictor per component (zero at scan start, also reset on RST).
    let mut prev_dc = [0i32; 4];
    // Restart interval bookkeeping.
    let restart_interval = headers.restart_interval as u32;
    let mut mcus_since_restart: u32 = 0;
    let mut expected_rst: u8 = 0;

    // Scratch buffers for one block's worth of work.
    let mut zz_coef = [0i16; 64];
    let mut nat_coef = [0i16; 64];

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // Restart handling.
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
                        // Decode this 8x8 block.
                        zz_coef.fill(0);
                        decode_block_baseline(
                            &mut br,
                            dc_tbl,
                            ac_tbl,
                            &mut prev_dc[scan_idx],
                            &mut zz_coef,
                        )?;

                        // De-zig-zag + dequantize into nat_coef.
                        for k in 0..64 {
                            nat_coef[ZIGZAG[k]] =
                                zz_coef[k].wrapping_mul(qt.values[ZIGZAG[k]] as i16);
                        }

                        // IDCT → 8x8 u8 block.
                        let mut block = [0u8; 64];
                        arch::backend::dct::idct_islow(&nat_coef, &mut block);

                        // Place into the plane.
                        let base_x = (mx * comp.h as u32 + h_block) as usize * 8;
                        let base_y = (my * comp.v as u32 + v_block) as usize * 8;
                        place_block(plane, base_x, base_y, &block);
                    }
                }
            }

            mcus_since_restart += 1;
        }
    }

    Ok((
        DecodedPlanes {
            width: frame.width as u32,
            height: frame.height as u32,
            components: planes,
        },
        br,
    ))
}

/// Decode the next 8x8 block in zig-zag order into `zz`. `prev_dc` is
/// updated to the new DC predictor on exit.
fn decode_block_baseline(
    br: &mut BitReader,
    dc_tbl: &HuffmanDecodeTable,
    ac_tbl: &HuffmanDecodeTable,
    prev_dc: &mut i32,
    zz: &mut [i16; 64],
) -> Result<()> {
    // ---- DC term ----
    let t = br.decode_symbol(dc_tbl)?;
    let dc_size = t as u32;
    let dc_diff = if dc_size == 0 {
        0
    } else {
        let bits = br.get_bits(dc_size)?;
        extend(bits, dc_size)
    };
    *prev_dc = prev_dc.wrapping_add(dc_diff);
    zz[0] = (*prev_dc) as i16;

    // ---- AC terms ----
    let mut k: usize = 1;
    while k < 64 {
        let rs = br.decode_symbol(ac_tbl)?;
        let run = (rs >> 4) as usize;
        let size = (rs & 0x0F) as u32;
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
        let bits = br.get_bits(size)?;
        zz[k] = extend(bits, size) as i16;
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

/// One slot per Huffman destination id (0..=3).
pub type HuffmanTableSlots = [Option<HuffmanDecodeTable>; 4];

/// Scan the supplied DHT specs into the `(dc, ac)` decode-table arrays
/// (indexed by destination id 0..=3). Used by the public decoder once
/// header parsing is complete.
pub fn build_huffman_tables(
    specs: &[super::markers::HuffmanTableSpec],
) -> Result<(HuffmanTableSlots, HuffmanTableSlots)> {
    let mut dc: [Option<HuffmanDecodeTable>; 4] = Default::default();
    let mut ac: [Option<HuffmanDecodeTable>; 4] = Default::default();
    for spec in specs {
        let tbl = HuffmanDecodeTable::from_spec(spec)?;
        let slot = match spec.class {
            0 => &mut dc[spec.id as usize],
            1 => &mut ac[spec.id as usize],
            _ => return Err(DecodeError::Malformed("DHT class > 1")),
        };
        *slot = Some(tbl);
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
