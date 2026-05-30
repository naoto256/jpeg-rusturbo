//! Progressive (SOF2) JPEG encoder.
//!
//! Splits the entropy-coded segment into multiple scans, each emitting
//! a sub-band of every block's coefficients (T.81 Annex G). The
//! encoder ships an eight-scan plan covering all four progressive
//! scan types from the spec, organized as one bit-plane pair
//! ("successive approximation" with `Al_first = 1`, `Al_refine = 0`,
//! `Ah_refine = 1`):
//!
//! 1. **DC interleaved first** — `Ss=0, Se=0, Ah=0, Al=1`
//! 2. **AC Y first**          — `Ss=1, Se=63, Ah=0, Al=1`
//! 3. **AC Cb first**         — `Ss=1, Se=63, Ah=0, Al=1`
//! 4. **AC Cr first**         — `Ss=1, Se=63, Ah=0, Al=1`
//! 5. **DC interleaved refine** — `Ss=0, Se=0, Ah=1, Al=0`
//! 6. **AC Y refine**          — `Ss=1, Se=63, Ah=1, Al=0`
//! 7. **AC Cb refine**         — `Ss=1, Se=63, Ah=1, Al=0`
//! 8. **AC Cr refine**         — `Ss=1, Se=63, Ah=1, Al=0`
//!
//! All four progressive scan-type encoders are implemented:
//!
//! - **DC first** uses arithmetic right shift (`(dc as i32) >> Al`) on
//!   the differential, so that the prior-scan reconstruction
//!   `V << Al` always rounds *toward negative infinity*. Refinement
//!   then OR's the LSB of the i16 bit-pattern into the running
//!   value — matching the decoder's `coef[0] |= 1 << al` semantics.
//! - **AC first** uses *toward-zero* signed division (`block[k] /
//!   (1 << Al)`) so the prior-scan reconstruction `V << Al` is the
//!   nearest multiple of `2^Al` *with smaller magnitude*. Refinement
//!   then ADDS `±(1 << Al)` (with the sign of the existing
//!   coefficient) to increase the magnitude. Different semantics
//!   from DC by design — the decoder's `refine_one_existing` does
//!   `v += sign * (1 << al)` rather than `v |= 1 << al`.
//! - **AC refine** is the most intricate of the four. Within each
//!   block: positions that were already significant in the prior
//!   scan get a single refinement bit; positions that become newly
//!   significant in this scan emit a `(zero_run, size=1)` Huffman
//!   symbol + sign bit; refinement bits for existing-significant
//!   positions interleaved with the new-significant emit are
//!   appended after the Huffman + sign (mirroring the decoder's
//!   walk-and-refine order). When a block has no newly-significant
//!   coefficients, its refinement bits accumulate into a deferred
//!   `eob_ref_bits` buffer that's emitted after the next EOBn
//!   Huffman code.
//!
//! Two emission strategies coexist:
//!
//! - **Default (Annex K standard tables)** — end-of-band is signalled
//!   with `EOB0` (symbol `0x00`) per block, because the Annex K
//!   reference AC Huffman tables don't carry `EOBn` (n ≥ 1) codes.
//!   This produces decodable output but pays a substantial size cost
//!   on natural content.
//! - **`set_optimize_huffman(true)` path** — runs a count-then-emit
//!   pass per scan, building per-scan custom Huffman tables that
//!   include `EOBn` codes, then emits multi-block `EOBn` runs via the
//!   shared [`EobrunState`]. The DHT segments are written immediately
//!   before each scan's SOS (= libjpeg-turbo's per-scan DHT
//!   convention).

use std::io::{self, Write};

use crate::tables::{
    STD_CHROMA_AC, STD_CHROMA_DC, STD_CHROMA_QUANT, STD_LUMA_AC, STD_LUMA_DC, STD_LUMA_QUANT,
    scale_quant_table,
};
use crate::{ChromaSubsampling, PixelLayout};

use super::huffman::{BitWriter, HuffmanTable, magnitude_category};
use super::huffman_optimize::{OptHuffmanSpec, build_optimal_huffman};
use super::markers;
use super::{JpegEncoder, SamplingScheme, Yuv420Scheme, Yuv422Scheme, Yuv444Scheme};
use crate::tables::Divisors;

/// Successive-approximation `Al` for first scans. `Al_refine = 0` (=
/// final precision) is hard-coded — a future minor could expose
/// multi-pair plans, but the decoder accepts any conforming
/// `Ah > Al ≥ 0` sequence today.
const AL_FIRST: u8 = 1;
const AL_REFINE: u8 = 0;
const AH_REFINE: u8 = 1;

/// Top-level progressive entry point — mirrors `encode_inner` for
/// the header / setup steps, then dispatches into the
/// scheme-monomorphized body so the per-subsampling constants are
/// compile-time.
pub(crate) fn encode_progressive_inner<W: Write>(
    enc: &mut JpegEncoder<W>,
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    div_luma: &Divisors,
    div_chroma: &Divisors,
) -> io::Result<()> {
    let (luma_q, chroma_q) = match enc.custom_quant() {
        Some((l, c)) => (*l, *c),
        None => (
            scale_quant_table(&STD_LUMA_QUANT, enc.quality()),
            scale_quant_table(&STD_CHROMA_QUANT, enc.quality()),
        ),
    };
    let optimize = enc.optimize();
    // Dispatch on subsampling so the body monomorphizes against the
    // per-scheme `MCU_W` / `Y_BLOCKS_PER_MCU` constants. Mirrors the
    // `dispatch_scheme!` macro pattern in `lib.rs`, inlined here so
    // we don't have to make the macro crate-public.
    match enc.subsampling() {
        ChromaSubsampling::Yuv444 => encode_progressive_scheme::<Yuv444Scheme, _>(
            enc, pixels, width, height, layout, &luma_q, &chroma_q, div_luma, div_chroma, optimize,
        ),
        ChromaSubsampling::Yuv422 => encode_progressive_scheme::<Yuv422Scheme, _>(
            enc, pixels, width, height, layout, &luma_q, &chroma_q, div_luma, div_chroma, optimize,
        ),
        ChromaSubsampling::Yuv420 => encode_progressive_scheme::<Yuv420Scheme, _>(
            enc, pixels, width, height, layout, &luma_q, &chroma_q, div_luma, div_chroma, optimize,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_progressive_scheme<S: SamplingScheme, W: Write>(
    enc: &mut JpegEncoder<W>,
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    luma_q: &[u8; 64],
    chroma_q: &[u8; 64],
    div_luma: &Divisors,
    div_chroma: &Divisors,
    optimize: bool,
) -> io::Result<()> {
    let mcus_x = width.div_ceil(S::MCU_W);
    let mcus_y = height.div_ceil(S::MCU_H);
    let total_mcus = (mcus_x as usize) * (mcus_y as usize);

    // ---- Quantize every block once.
    //
    // Progressive emits the *same* coefficients across multiple
    // scans, so we materialize all blocks up front and let each scan
    // read from the buffer. Storage is `≈ 2 bytes/pixel × subsampling
    // factor` — a 4K frame fits in ~13 MB.
    let mut y_blocks: Vec<[i16; 64]> = Vec::with_capacity(total_mcus * S::Y_BLOCKS_PER_MCU);
    let mut cb_blocks: Vec<[i16; 64]> = Vec::with_capacity(total_mcus);
    let mut cr_blocks: Vec<[i16; 64]> = Vec::with_capacity(total_mcus);
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            S::quantize_one_mcu_per_comp(
                pixels,
                width,
                height,
                layout,
                mx,
                my,
                div_luma,
                div_chroma,
                &mut y_blocks,
                &mut cb_blocks,
                &mut cr_blocks,
            );
        }
    }

    // ---- Build Y raster order.
    //
    // Y blocks live in `y_blocks` in MCU-chunked order: each MCU
    // contributes `H_V.0 × H_V.1` blocks row-major within the MCU,
    // followed by the next MCU. The interleaved DC scan reads in
    // exactly that order (= what the decoder's interleaved MCU walk
    // expects), so we hand the raw slice in.
    //
    // The AC scans, however, are **non-interleaved** — the decoder
    // walks one component's blocks in `blocks_x × blocks_y` raster
    // order (T.81 G.1.2 / A.2.2). For Cb / Cr at every supported
    // subsampling each MCU contributes exactly one block, so the
    // MCU-chunked layout already matches raster. For Y at 4:2:2 /
    // 4:2:0 it doesn't (e.g. 4:2:0 MCU(1,0)'s top-left block is at
    // raster `(bx=2, by=0)`, not `(bx=2, by=1)` which is where the
    // MCU-chunked stride would land it). Reorder once into a raster
    // index permutation; the AC scans iterate that.
    let y_raster_indices: Vec<usize> = {
        let (h_y, v_y) = S::H_V;
        let h_y = h_y as usize;
        let v_y = v_y as usize;
        let blocks_x = mcus_x as usize * h_y;
        let mut out = vec![0usize; y_blocks.len()];
        for my in 0..mcus_y as usize {
            for mx in 0..mcus_x as usize {
                let mcu_idx = my * mcus_x as usize + mx;
                for sv in 0..v_y {
                    for sh in 0..h_y {
                        let bx = mx * h_y + sh;
                        let by = my * v_y + sv;
                        let mcu_intra = sv * h_y + sh;
                        let raster_idx = by * blocks_x + bx;
                        out[raster_idx] = mcu_idx * (h_y * v_y) + mcu_intra;
                    }
                }
            }
        }
        out
    };

    // ---- Header emission.
    let exif_blob: Option<Vec<u8>> = enc.exif_bytes().map(<[u8]>::to_vec);
    let icc_blob: Option<Vec<u8>> = enc.icc_bytes().map(<[u8]>::to_vec);
    markers::write_soi(enc.out_mut())?;
    markers::write_app0_jfif(enc.out_mut())?;
    if let Some(exif) = &exif_blob {
        markers::write_app1_exif(enc.out_mut(), exif)?;
    }
    if let Some(icc) = &icc_blob {
        markers::write_app2_icc(enc.out_mut(), icc)?;
    }
    markers::write_dqt(enc.out_mut(), 0, luma_q)?;
    markers::write_dqt(enc.out_mut(), 1, chroma_q)?;

    let (h_y, v_y) = S::H_V;
    markers::write_sof2(
        enc.out_mut(),
        width as u16,
        height as u16,
        &[(1, h_y, v_y, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
    )?;

    if optimize {
        encode_progressive_scans_optimized::<S, _>(
            enc.out_mut(),
            &y_blocks,
            &cb_blocks,
            &cr_blocks,
            &y_raster_indices,
        )?;
    } else {
        encode_progressive_scans_standard::<S, _>(
            enc.out_mut(),
            &y_blocks,
            &cb_blocks,
            &cr_blocks,
            &y_raster_indices,
        )?;
    }

    markers::write_eoi(enc.out_mut())?;
    Ok(())
}

/// Standard (Annex K reference tables) progressive scan plan. Emits
/// the four DHT segments up front, then runs the eight scans with the
/// per-block `EOB0` strategy — byte-identical to the 0.8.0 progressive
/// output.
fn encode_progressive_scans_standard<S: SamplingScheme, W: Write>(
    out: &mut W,
    y_blocks: &[[i16; 64]],
    cb_blocks: &[[i16; 64]],
    cr_blocks: &[[i16; 64]],
    y_raster_indices: &[usize],
) -> io::Result<()> {
    let dc_luma = HuffmanTable::from_std(&STD_LUMA_DC);
    let ac_luma = HuffmanTable::from_std(&STD_LUMA_AC);
    let dc_chroma = HuffmanTable::from_std(&STD_CHROMA_DC);
    let ac_chroma = HuffmanTable::from_std(&STD_CHROMA_AC);
    markers::write_dht(out, 0, 0, &STD_LUMA_DC)?;
    markers::write_dht(out, 1, 0, &STD_LUMA_AC)?;
    markers::write_dht(out, 0, 1, &STD_CHROMA_DC)?;
    markers::write_dht(out, 1, 1, &STD_CHROMA_AC)?;

    // Scan 1: DC interleaved first.
    encode_dc_interleaved_first::<S, _>(
        out, y_blocks, cb_blocks, cr_blocks, &dc_luma, &dc_chroma, AL_FIRST,
    )?;

    // Scans 2-4: AC first per component. Standard tables → no EOBn,
    // we pass `allow_eobn = false` so each block self-terminates.
    encode_ac_first_scan_indexed(
        out,
        y_blocks,
        y_raster_indices,
        &ac_luma,
        1,
        0,
        1,
        63,
        AL_FIRST,
        false,
    )?;
    encode_ac_first_scan(out, cb_blocks, &ac_chroma, 2, 1, 1, 63, AL_FIRST, false)?;
    encode_ac_first_scan(out, cr_blocks, &ac_chroma, 3, 1, 1, 63, AL_FIRST, false)?;

    // Scan 5: DC interleaved refine.
    encode_dc_interleaved_refine::<S, _>(
        out, y_blocks, cb_blocks, cr_blocks, AH_REFINE, AL_REFINE,
    )?;

    // Scans 6-8: AC refine per component.
    encode_ac_refine_scan_indexed(
        out,
        y_blocks,
        y_raster_indices,
        &ac_luma,
        1,
        0,
        1,
        63,
        AH_REFINE,
        AL_REFINE,
        false,
    )?;
    encode_ac_refine_scan(
        out, cb_blocks, &ac_chroma, 2, 1, 1, 63, AH_REFINE, AL_REFINE, false,
    )?;
    encode_ac_refine_scan(
        out, cr_blocks, &ac_chroma, 3, 1, 1, 63, AH_REFINE, AL_REFINE, false,
    )?;
    Ok(())
}

/// Two-pass optimized-Huffman progressive scan plan. For each scan:
/// (1) count symbol frequencies under the EOBn-aware emission strategy,
/// (2) build optimal canonical Huffman tables from those counts,
/// (3) emit the DHT segments immediately before the scan's SOS,
/// (4) re-walk the blocks emitting bits under the *same* strategy.
fn encode_progressive_scans_optimized<S: SamplingScheme, W: Write>(
    out: &mut W,
    y_blocks: &[[i16; 64]],
    cb_blocks: &[[i16; 64]],
    cr_blocks: &[[i16; 64]],
    y_raster_indices: &[usize],
) -> io::Result<()> {
    // ---- Scan 1: DC interleaved first.
    let (dc_l_freq, dc_c_freq) = count_dc_interleaved_first::<S>(y_blocks, cb_blocks, cr_blocks);
    let dc_l_spec = build_dc_table(&dc_l_freq, /*luma=*/ true);
    let dc_c_spec = build_dc_table(&dc_c_freq, /*luma=*/ false);
    markers::write_dht_bits_values(out, 0, 0, &dc_l_spec.bits, &dc_l_spec.values)?;
    markers::write_dht_bits_values(out, 0, 1, &dc_c_spec.bits, &dc_c_spec.values)?;
    let dc_l_tab = HuffmanTable::from_bits_values(&dc_l_spec.bits, &dc_l_spec.values);
    let dc_c_tab = HuffmanTable::from_bits_values(&dc_c_spec.bits, &dc_c_spec.values);
    encode_dc_interleaved_first::<S, _>(
        out, y_blocks, cb_blocks, cr_blocks, &dc_l_tab, &dc_c_tab, AL_FIRST,
    )?;

    // ---- Scans 2-4: AC first per component. Each scan gets its own
    // counted frequencies (the EOBn strategy means the symbol stream
    // depends on the run distribution, which differs per component).
    encode_one_ac_first_scan(
        out,
        y_blocks,
        Some(y_raster_indices),
        1,
        0,
        1,
        63,
        AL_FIRST,
        /*luma=*/ true,
    )?;
    encode_one_ac_first_scan(
        out, cb_blocks, None, 2, 1, 1, 63, AL_FIRST, /*luma=*/ false,
    )?;
    encode_one_ac_first_scan(
        out, cr_blocks, None, 3, 1, 1, 63, AL_FIRST, /*luma=*/ false,
    )?;

    // ---- Scan 5: DC interleaved refine. Pure raw bits, no Huffman
    // — no DHT needed and nothing to optimize.
    encode_dc_interleaved_refine::<S, _>(
        out, y_blocks, cb_blocks, cr_blocks, AH_REFINE, AL_REFINE,
    )?;

    // ---- Scans 6-8: AC refine per component.
    encode_one_ac_refine_scan(
        out,
        y_blocks,
        Some(y_raster_indices),
        1,
        0,
        1,
        63,
        AH_REFINE,
        AL_REFINE,
        true,
    )?;
    encode_one_ac_refine_scan(
        out, cb_blocks, None, 2, 1, 1, 63, AH_REFINE, AL_REFINE, false,
    )?;
    encode_one_ac_refine_scan(
        out, cr_blocks, None, 3, 1, 1, 63, AH_REFINE, AL_REFINE, false,
    )?;
    Ok(())
}

/// Build an optimal DC table, picking the matching Annex K fallback so
/// an all-zero histogram (theoretically possible on a degenerate input)
/// still emits a valid DHT.
fn build_dc_table(freq: &[u32; 257], luma: bool) -> OptHuffmanSpec {
    if luma {
        build_optimal_huffman(freq, &STD_LUMA_DC.bits, STD_LUMA_DC.values)
    } else {
        build_optimal_huffman(freq, &STD_CHROMA_DC.bits, STD_CHROMA_DC.values)
    }
}

fn build_ac_table(freq: &[u32; 257], luma: bool) -> OptHuffmanSpec {
    if luma {
        build_optimal_huffman(freq, &STD_LUMA_AC.bits, STD_LUMA_AC.values)
    } else {
        build_optimal_huffman(freq, &STD_CHROMA_AC.bits, STD_CHROMA_AC.values)
    }
}

// ============================================================================
// DC scans
// ============================================================================

/// SOS + entropy for the interleaved DC-first scan
/// (`Ss=0, Se=0, Ah=0, Al`).
fn encode_dc_interleaved_first<S: SamplingScheme, W: Write>(
    out: &mut W,
    y_blocks: &[[i16; 64]],
    cb_blocks: &[[i16; 64]],
    cr_blocks: &[[i16; 64]],
    dc_luma: &HuffmanTable,
    dc_chroma: &HuffmanTable,
    al: u8,
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(1, 0, 0), (2, 1, 0), (3, 1, 0)], 0, 0, 0, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(y_blocks.len() * 4);
    let (mut prev_y, mut prev_cb, mut prev_cr) = (0i32, 0i32, 0i32);
    let y_mcus = y_blocks.chunks_exact(S::Y_BLOCKS_PER_MCU);
    for (y_chunk, (cb, cr)) in y_mcus.zip(cb_blocks.iter().zip(cr_blocks.iter())) {
        for y in y_chunk {
            prev_y = encode_dc_first(&mut bw, y[0], prev_y, dc_luma, al)?;
        }
        prev_cb = encode_dc_first(&mut bw, cb[0], prev_cb, dc_chroma, al)?;
        prev_cr = encode_dc_first(&mut bw, cr[0], prev_cr, dc_chroma, al)?;
    }
    bw.flush_to_byte_boundary()
}

/// Frequency-counting twin of [`encode_dc_interleaved_first`]. Walks
/// the blocks in the same order and bumps the (luma, chroma) DC size
/// histograms instead of emitting bits.
fn count_dc_interleaved_first<S: SamplingScheme>(
    y_blocks: &[[i16; 64]],
    cb_blocks: &[[i16; 64]],
    cr_blocks: &[[i16; 64]],
) -> ([u32; 257], [u32; 257]) {
    let mut luma_freq = [0u32; 257];
    let mut chroma_freq = [0u32; 257];
    let (mut prev_y, mut prev_cb, mut prev_cr) = (0i32, 0i32, 0i32);
    let y_mcus = y_blocks.chunks_exact(S::Y_BLOCKS_PER_MCU);
    for (y_chunk, (cb, cr)) in y_mcus.zip(cb_blocks.iter().zip(cr_blocks.iter())) {
        for y in y_chunk {
            prev_y = count_dc_first(y[0], prev_y, &mut luma_freq, AL_FIRST);
        }
        prev_cb = count_dc_first(cb[0], prev_cb, &mut chroma_freq, AL_FIRST);
        prev_cr = count_dc_first(cr[0], prev_cr, &mut chroma_freq, AL_FIRST);
    }
    (luma_freq, chroma_freq)
}

/// SOS + entropy for the interleaved DC-refine scan
/// (`Ss=0, Se=0, Ah, Al`). One raw bit per block per component
/// (no Huffman in this scan type).
fn encode_dc_interleaved_refine<S: SamplingScheme, W: Write>(
    out: &mut W,
    y_blocks: &[[i16; 64]],
    cb_blocks: &[[i16; 64]],
    cr_blocks: &[[i16; 64]],
    ah: u8,
    al: u8,
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(1, 0, 0), (2, 1, 0), (3, 1, 0)], 0, 0, ah, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(y_blocks.len() / 2);
    let y_mcus = y_blocks.chunks_exact(S::Y_BLOCKS_PER_MCU);
    for (y_chunk, (cb, cr)) in y_mcus.zip(cb_blocks.iter().zip(cr_blocks.iter())) {
        for y in y_chunk {
            emit_dc_refine_bit(&mut bw, y[0], al)?;
        }
        emit_dc_refine_bit(&mut bw, cb[0], al)?;
        emit_dc_refine_bit(&mut bw, cr[0], al)?;
    }
    bw.flush_to_byte_boundary()
}

/// Emit a DC-first-scan symbol for one block. The differential is
/// computed on the **arithmetically right-shifted** DC value — that
/// way the decoder's `V << Al` reconstruction rounds toward negative
/// infinity, leaving the low `Al` bits of the original DC available
/// for the OR-based refine path.
///
/// `prev_dc_shifted` carries the shifted DC predictor across blocks
/// in the same scan; the next predictor is the just-emitted shifted
/// DC.
fn encode_dc_first<W: Write>(
    bw: &mut BitWriter<W>,
    dc: i16,
    prev_dc_shifted: i32,
    dc_tab: &HuffmanTable,
    al: u8,
) -> io::Result<i32> {
    let dc_shifted = (dc as i32) >> al;
    let diff = dc_shifted - prev_dc_shifted;
    let (size, bits) = magnitude_category(diff);
    let entry = dc_tab.packed[size as usize];
    let code = entry & 0xFFFF;
    let len = entry >> 16;
    // Fused (Huffman code, magnitude bits) write.
    bw.write_bits((code << size) | bits, len + size as u32)?;
    Ok(dc_shifted)
}

/// Frequency-counting twin of [`encode_dc_first`]. Increments
/// `dc_freq[size]` for the magnitude category that would be emitted
/// and returns the next predictor.
fn count_dc_first(dc: i16, prev_dc_shifted: i32, dc_freq: &mut [u32; 257], al: u8) -> i32 {
    let dc_shifted = (dc as i32) >> al;
    let diff = dc_shifted - prev_dc_shifted;
    let (size, _) = magnitude_category(diff);
    dc_freq[size as usize] += 1;
    dc_shifted
}

/// Emit a one-bit DC refinement: the bit at position `al` of the
/// i16 two's-complement representation. The decoder OR's this bit
/// into `coef[0]`, so it must be `(dc as u16 >> al) & 1` rather than
/// `(|dc| >> al) & 1` — the two diverge for `al > 0` with negative
/// `dc` (e.g. `dc = -3`, `al = 1` → bit pattern says 0, magnitude
/// says 1). At `al = 0` (the only refine `Al` in this version)
/// the formulas agree, but writing it out keeps the function correct
/// for future multi-pair scan plans.
fn emit_dc_refine_bit<W: Write>(bw: &mut BitWriter<W>, dc: i16, al: u8) -> io::Result<()> {
    let bit = ((dc as u16) >> al) & 1;
    bw.write_bits(bit as u32, 1)
}

// ============================================================================
// EOBRUN encoding strategy
// ============================================================================
//
// `EOBn` symbols carry a run-length encoding of "end of band" terminators
// across multiple consecutive blocks. The decoder reads symbol byte
// `(N << 4)` for `N ∈ 0..=14`, then `N` extra unsigned bits, and adds
// `(1 << N) + extra` to its block-skip counter. The encoder mirrors
// this exactly:
//
// * `EobrunState::run` tracks the count of consecutive blocks (so far in
//   the scan) whose contribution would be "end of band, nothing new".
//   Pre-flush of any block that emits a real symbol — and at scan end —
//   the run is converted into one or more `EOBn + extra-bits` emissions.
// * For the AC-refine scan, blocks inside an EOBn run still need to
//   carry their *correction bits* for already-significant coefficients.
//   Those bits are buffered in `EobrunState::pending_ref_bits` while the
//   run accumulates and are written out **immediately after** the EOBn
//   header and its extra bits — matching the decoder's
//   `refine_existing_band` walk on each block of the run.
//
// The same struct + `flush` / `extend` API is used by both the counting
// pass (`Sink::count`) and the emit pass (`Sink::emit`) — only the
// `Sink::ac_symbol` / `Sink::raw_bits` implementations differ.

/// Carries the EOBn run-length plus any deferred AC-refine correction
/// bits accumulated since the run started. A `flush` writes one or
/// more `EOBn + extra-bits` token(s), drains the correction-bit buffer,
/// and resets the run to zero.
#[derive(Default)]
struct EobrunState {
    /// Number of consecutive blocks whose contribution has been
    /// deferred into the run.
    run: u32,
    /// AC-refine correction bits accumulated while the run grew. Each
    /// entry is `(value, n_bits)`. Empty for AC-first scans (no
    /// correction bits there).
    pending_ref_bits: Vec<(u32, u32)>,
}

impl EobrunState {
    fn extend(&mut self) {
        self.run += 1;
    }

    fn push_ref_bit(&mut self, bit: u32) {
        self.pending_ref_bits.push((bit, 1));
    }
}

/// Sink trait shared by the counting and emit passes. Each AC scan
/// instantiates one of these; the per-scan walker calls into it
/// without knowing whether bits are landing in a histogram or in the
/// bit stream.
trait Sink {
    /// Account for / emit an AC symbol byte (`(run << 4) | size`).
    fn ac_symbol(&mut self, sym: u8) -> io::Result<()>;
    /// Account for / emit raw payload bits (magnitudes, signs,
    /// EOBn extras, AC-refine correction bits). Counting impl is a
    /// no-op; emitting impl forwards to the bit writer.
    fn raw_bits(&mut self, value: u32, n_bits: u32) -> io::Result<()>;
}

/// Counting sink: just bumps `ac_freq[sym]`.
struct CountSink<'a> {
    ac_freq: &'a mut [u32; 257],
}
impl Sink for CountSink<'_> {
    #[inline]
    fn ac_symbol(&mut self, sym: u8) -> io::Result<()> {
        self.ac_freq[sym as usize] += 1;
        Ok(())
    }
    #[inline]
    fn raw_bits(&mut self, _value: u32, _n_bits: u32) -> io::Result<()> {
        Ok(())
    }
}

/// Emit sink: writes Huffman + raw bits to the bit stream.
struct EmitSink<'a, W: Write> {
    bw: &'a mut BitWriter<W>,
    ac_tab: &'a HuffmanTable,
}
impl<W: Write> Sink for EmitSink<'_, W> {
    #[inline]
    fn ac_symbol(&mut self, sym: u8) -> io::Result<()> {
        let entry = self.ac_tab.packed[sym as usize];
        let code = entry & 0xFFFF;
        let len = entry >> 16;
        debug_assert!(len > 0, "AC symbol {sym:#04x} has no code in optimal table");
        self.bw.write_bits(code, len)
    }
    #[inline]
    fn raw_bits(&mut self, value: u32, n_bits: u32) -> io::Result<()> {
        self.bw.write_bits(value, n_bits)
    }
}

/// Flush the accumulated EOBn run as one or more
/// `EOBn + extra-bits (+ buffered correction bits)` emissions. After
/// the call `state.run == 0` and `state.pending_ref_bits` is empty.
///
/// Strategy mirrors libjpeg-turbo's `emit_eobrun` in `jcphuff.c`: pick
/// the largest `N ∈ 0..=14` with `(1 << N) ≤ run`, emit symbol `N<<4`
/// followed by `N` extra bits `run - (1 << N)`. One emission covers
/// runs up to `2^15 - 1 = 32767`; the outer loop handles larger runs
/// by clamping at `N = 14` repeatedly.
fn flush_eobrun(sink: &mut dyn Sink, state: &mut EobrunState) -> io::Result<()> {
    while state.run > 0 {
        // Largest N ≤ 14 with (1 << N) ≤ run.
        let mut n: u32 = 0;
        while n < 14 && (1u32 << (n + 1)) <= state.run {
            n += 1;
        }
        let base = 1u32 << n;
        // One EOBn-N encodes at most `base + (base - 1) = 2^(N+1) - 1`
        // blocks. Cap the extra at the N-bit field width; the outer
        // loop emits another token for any residual.
        let max_extra = base - 1;
        let extra = (state.run - base).min(max_extra);
        let sym = (n as u8) << 4;
        sink.ac_symbol(sym)?;
        if n > 0 {
            sink.raw_bits(extra, n)?;
        }
        state.run -= base + extra;
    }
    // Drain deferred AC-refine correction bits in walk order.
    for (val, n) in state.pending_ref_bits.drain(..) {
        sink.raw_bits(val, n)?;
    }
    Ok(())
}

// ============================================================================
// AC first scan
// ============================================================================

/// One-shot count + build + DHT + emit for a single AC-first scan
/// (optimize path). `raster_indices` permutes `blocks` into raster
/// order for Y at subsampled layouts; pass `None` for chroma (already
/// raster).
#[allow(clippy::too_many_arguments)]
fn encode_one_ac_first_scan<W: Write>(
    out: &mut W,
    blocks: &[[i16; 64]],
    raster_indices: Option<&[usize]>,
    component_id: u8,
    ac_tab_id: u8,
    ss: u8,
    se: u8,
    al: u8,
    luma: bool,
) -> io::Result<()> {
    // Counting pass.
    let mut freq = [0u32; 257];
    {
        let mut sink = CountSink { ac_freq: &mut freq };
        let mut state = EobrunState::default();
        walk_ac_first(blocks, raster_indices, ss, se, al, &mut sink, &mut state)?;
        flush_eobrun(&mut sink, &mut state)?;
    }
    let spec = build_ac_table(&freq, luma);
    markers::write_dht_bits_values(out, 1, ac_tab_id, &spec.bits, &spec.values)?;
    let ac_tab = HuffmanTable::from_bits_values(&spec.bits, &spec.values);
    // Emit pass.
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, 0, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 8);
    {
        let mut sink = EmitSink {
            bw: &mut bw,
            ac_tab: &ac_tab,
        };
        let mut state = EobrunState::default();
        walk_ac_first(blocks, raster_indices, ss, se, al, &mut sink, &mut state)?;
        flush_eobrun(&mut sink, &mut state)?;
    }
    bw.flush_to_byte_boundary()
}

/// Standard-tables variant: walk blocks in scan order and emit
/// per-block contributions with `allow_eobn = false` so end-of-band is
/// always signalled as `EOB0` (no run extension). This is what keeps
/// the default progressive output byte-identical to 0.8.0.
#[allow(clippy::too_many_arguments)]
fn encode_ac_first_scan_indexed<W: Write>(
    out: &mut W,
    blocks: &[[i16; 64]],
    raster_indices: &[usize],
    ac_tab: &HuffmanTable,
    component_id: u8,
    ac_tab_id: u8,
    ss: u8,
    se: u8,
    al: u8,
    allow_eobn: bool,
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, 0, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 8);
    if allow_eobn {
        let mut sink = EmitSink {
            bw: &mut bw,
            ac_tab,
        };
        let mut state = EobrunState::default();
        walk_ac_first(
            blocks,
            Some(raster_indices),
            ss,
            se,
            al,
            &mut sink,
            &mut state,
        )?;
        flush_eobrun(&mut sink, &mut state)?;
    } else {
        for &idx in raster_indices {
            encode_ac_first_eob0(&mut bw, &blocks[idx], ss, se, ac_tab, al)?;
        }
    }
    bw.flush_to_byte_boundary()
}

#[allow(clippy::too_many_arguments)]
fn encode_ac_first_scan<W: Write>(
    out: &mut W,
    blocks: &[[i16; 64]],
    ac_tab: &HuffmanTable,
    component_id: u8,
    ac_tab_id: u8,
    ss: u8,
    se: u8,
    al: u8,
    allow_eobn: bool,
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, 0, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 8);
    if allow_eobn {
        let mut sink = EmitSink {
            bw: &mut bw,
            ac_tab,
        };
        let mut state = EobrunState::default();
        walk_ac_first(blocks, None, ss, se, al, &mut sink, &mut state)?;
        flush_eobrun(&mut sink, &mut state)?;
    } else {
        for block in blocks {
            encode_ac_first_eob0(&mut bw, block, ss, se, ac_tab, al)?;
        }
    }
    bw.flush_to_byte_boundary()
}

/// Shared AC-first walk used by both count + emit (EOBn-aware) passes.
/// Iterates blocks in scan order, accumulating "all-zero band" blocks
/// into `state.run` and flushing the run before any block that needs
/// to emit real symbols.
fn walk_ac_first(
    blocks: &[[i16; 64]],
    raster_indices: Option<&[usize]>,
    ss: u8,
    se: u8,
    al: u8,
    sink: &mut dyn Sink,
    state: &mut EobrunState,
) -> io::Result<()> {
    let n = blocks.len();
    for idx in 0..n {
        let block_idx = raster_indices.map(|r| r[idx]).unwrap_or(idx);
        let block = &blocks[block_idx];
        ac_first_one_block(block, ss, se, al, sink, state)?;
    }
    Ok(())
}

/// One block's contribution to an AC-first scan under the EOBn-aware
/// strategy. If the block's band is entirely zero (after toward-zero
/// shift by `al`), it extends `state.run` and emits nothing.
/// Otherwise the pending EOBn run is flushed first, then run/size
/// symbols + magnitude bits are emitted in zig-zag order; a trailing
/// run of zeros extends `state.run` (i.e. the block "ends with an
/// EOB") for the next iteration to pack.
fn ac_first_one_block(
    block: &[i16; 64],
    ss: u8,
    se: u8,
    al: u8,
    sink: &mut dyn Sink,
    state: &mut EobrunState,
) -> io::Result<()> {
    let ss = ss as usize;
    let se = se as usize;
    // Toward-zero shift; see module docstring for rationale.
    let shifted = |k: usize| -> i32 {
        let coef = block[k];
        let abs = (coef.unsigned_abs() >> al) as i32;
        if coef < 0 { -abs } else { abs }
    };
    let mut last_nz: Option<usize> = None;
    for k in ss..=se {
        if shifted(k) != 0 {
            last_nz = Some(k);
        }
    }
    let last_nz = match last_nz {
        None => {
            state.extend();
            return Ok(());
        }
        Some(k) => k,
    };
    // We have real symbols to emit — flush any pending EOBn run first.
    flush_eobrun(sink, state)?;
    let mut zero_run: u32 = 0;
    for k in ss..=last_nz {
        let v = shifted(k);
        if v == 0 {
            zero_run += 1;
            continue;
        }
        while zero_run >= 16 {
            sink.ac_symbol(0xF0)?;
            zero_run -= 16;
        }
        let (size, bits) = magnitude_category(v);
        debug_assert!(size <= 10, "AC magnitude category {size} > 10");
        let symbol = ((zero_run as u8) << 4) | (size & 0x0F);
        sink.ac_symbol(symbol)?;
        sink.raw_bits(bits, size as u32)?;
        zero_run = 0;
    }
    if last_nz < se {
        // Trailing zeros in this block contribute one EOB.
        state.extend();
    }
    Ok(())
}

/// Standard-tables (no EOBn) AC-first per-block emitter. Mirrors the
/// 0.8.0 behavior exactly so the byte stream is unchanged when
/// `allow_eobn = false`.
fn encode_ac_first_eob0<W: Write>(
    bw: &mut BitWriter<W>,
    block: &[i16; 64],
    ss: u8,
    se: u8,
    ac_tab: &HuffmanTable,
    al: u8,
) -> io::Result<()> {
    let ss = ss as usize;
    let se = se as usize;
    let shifted = |k: usize| -> i32 {
        let coef = block[k];
        let abs = (coef.unsigned_abs() >> al) as i32;
        if coef < 0 { -abs } else { abs }
    };
    let mut last_nz: Option<usize> = None;
    for k in ss..=se {
        if shifted(k) != 0 {
            last_nz = Some(k);
        }
    }
    let last_nz = match last_nz {
        None => {
            emit_eob0(bw, ac_tab)?;
            return Ok(());
        }
        Some(k) => k,
    };
    let mut zero_run: u32 = 0;
    for k in ss..=last_nz {
        let v = shifted(k);
        if v == 0 {
            zero_run += 1;
            continue;
        }
        while zero_run >= 16 {
            let zrl = ac_tab.packed[0xF0];
            bw.write_bits(zrl & 0xFFFF, zrl >> 16)?;
            zero_run -= 16;
        }
        let (size, bits) = magnitude_category(v);
        debug_assert!(size <= 10, "AC magnitude category {size} > 10");
        let symbol = ((zero_run as usize) << 4) | (size as usize & 0x0F);
        let entry = ac_tab.packed[symbol];
        let code = entry & 0xFFFF;
        let len = entry >> 16;
        bw.write_bits((code << size) | bits, len + size as u32)?;
        zero_run = 0;
    }
    if last_nz < se {
        emit_eob0(bw, ac_tab)?;
    }
    Ok(())
}

/// Emit a single `EOB0` Huffman symbol (symbol `0x00`, "end of band,
/// no run extension"). Used only by the standard-tables (no-EOBn)
/// path; the optimize path goes through [`flush_eobrun`].
fn emit_eob0<W: Write>(bw: &mut BitWriter<W>, ac_tab: &HuffmanTable) -> io::Result<()> {
    let entry = ac_tab.packed[0x00];
    let code = entry & 0xFFFF;
    let len = entry >> 16;
    bw.write_bits(code, len)
}

// ============================================================================
// AC refine scan
// ============================================================================

/// One-shot count + build + DHT + emit for a single AC-refine scan
/// (optimize path).
#[allow(clippy::too_many_arguments)]
fn encode_one_ac_refine_scan<W: Write>(
    out: &mut W,
    blocks: &[[i16; 64]],
    raster_indices: Option<&[usize]>,
    component_id: u8,
    ac_tab_id: u8,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    luma: bool,
) -> io::Result<()> {
    let mut freq = [0u32; 257];
    {
        let mut sink = CountSink { ac_freq: &mut freq };
        let mut state = EobrunState::default();
        walk_ac_refine(
            blocks,
            raster_indices,
            ss,
            se,
            ah,
            al,
            &mut sink,
            &mut state,
        )?;
        flush_eobrun(&mut sink, &mut state)?;
    }
    let spec = build_ac_table(&freq, luma);
    markers::write_dht_bits_values(out, 1, ac_tab_id, &spec.bits, &spec.values)?;
    let ac_tab = HuffmanTable::from_bits_values(&spec.bits, &spec.values);
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, ah, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 4);
    {
        let mut sink = EmitSink {
            bw: &mut bw,
            ac_tab: &ac_tab,
        };
        let mut state = EobrunState::default();
        walk_ac_refine(
            blocks,
            raster_indices,
            ss,
            se,
            ah,
            al,
            &mut sink,
            &mut state,
        )?;
        flush_eobrun(&mut sink, &mut state)?;
    }
    bw.flush_to_byte_boundary()
}

/// Standard-tables variants: keep the per-block `EOB0` strategy.
#[allow(clippy::too_many_arguments)]
fn encode_ac_refine_scan_indexed<W: Write>(
    out: &mut W,
    blocks: &[[i16; 64]],
    raster_indices: &[usize],
    ac_tab: &HuffmanTable,
    component_id: u8,
    ac_tab_id: u8,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    allow_eobn: bool,
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, ah, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 4);
    if allow_eobn {
        let mut sink = EmitSink {
            bw: &mut bw,
            ac_tab,
        };
        let mut state = EobrunState::default();
        walk_ac_refine(
            blocks,
            Some(raster_indices),
            ss,
            se,
            ah,
            al,
            &mut sink,
            &mut state,
        )?;
        flush_eobrun(&mut sink, &mut state)?;
    } else {
        for &idx in raster_indices {
            encode_ac_refine_block_eob0(&mut bw, &blocks[idx], ss, se, ah, al, ac_tab)?;
        }
    }
    bw.flush_to_byte_boundary()
}

#[allow(clippy::too_many_arguments)]
fn encode_ac_refine_scan<W: Write>(
    out: &mut W,
    blocks: &[[i16; 64]],
    ac_tab: &HuffmanTable,
    component_id: u8,
    ac_tab_id: u8,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    allow_eobn: bool,
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, ah, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 4);
    if allow_eobn {
        let mut sink = EmitSink {
            bw: &mut bw,
            ac_tab,
        };
        let mut state = EobrunState::default();
        walk_ac_refine(blocks, None, ss, se, ah, al, &mut sink, &mut state)?;
        flush_eobrun(&mut sink, &mut state)?;
    } else {
        for block in blocks {
            encode_ac_refine_block_eob0(&mut bw, block, ss, se, ah, al, ac_tab)?;
        }
    }
    bw.flush_to_byte_boundary()
}

#[allow(clippy::too_many_arguments)]
fn walk_ac_refine(
    blocks: &[[i16; 64]],
    raster_indices: Option<&[usize]>,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    sink: &mut dyn Sink,
    state: &mut EobrunState,
) -> io::Result<()> {
    let n = blocks.len();
    for idx in 0..n {
        let block_idx = raster_indices.map(|r| r[idx]).unwrap_or(idx);
        let block = &blocks[block_idx];
        ac_refine_one_block(block, ss, se, ah, al, sink, state)?;
    }
    Ok(())
}

/// One block's contribution to an AC-refine scan under the EOBn-aware
/// strategy. Mirrors `encode_ac_refine_block_eob0` but defers
/// "no-new-significant" blocks into `state` for run-packing.
///
/// Critical invariant for cross-decoder compatibility: when a block
/// has new-significants, any pending EOBn run is flushed **first**
/// (carrying the correction bits of prior all-existing-only blocks),
/// and then the new block's symbols are emitted normally; refinement
/// bits for *this* block's existing-sigs are interleaved with the new-
/// sig emissions as before. Trailing existing-sigs after the last new-
/// sig in *this* block extend the run (their correction bits queue
/// into `state.pending_ref_bits`, awaiting the next flush).
#[allow(clippy::needless_range_loop)]
fn ac_refine_one_block(
    block: &[i16; 64],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    sink: &mut dyn Sink,
    state: &mut EobrunState,
) -> io::Result<()> {
    let ss = ss as usize;
    let se = se as usize;
    let prev_threshold = 1u16 << ah;
    let prev_sig = |k: usize| block[k].unsigned_abs() >= prev_threshold;
    let new_sig = |k: usize| {
        let av = block[k].unsigned_abs();
        av < prev_threshold && (av >> al) != 0
    };

    let has_new = (ss..=se).any(new_sig);

    if !has_new {
        // No new-sig in this block → extend the EOBn run, queue this
        // block's existing-sig correction bits for output after the
        // eventual EOBn.
        state.extend();
        for k in ss..=se {
            if prev_sig(k) {
                let bit = ((block[k].unsigned_abs() >> al) & 1) as u32;
                state.push_ref_bit(bit);
            }
        }
        return Ok(());
    }

    // This block emits real symbols → flush any pending run first.
    // After flush, `state.run == 0` and `pending_ref_bits` is empty;
    // refinement bits for *this* block's existing-sigs are interleaved
    // inline (NOT queued, since they belong with this block's
    // emissions, not the run prelude).
    flush_eobrun(sink, state)?;

    let mut sub_k = ss;
    let mut k = ss;
    while k <= se {
        if new_sig(k) {
            let mut zero_count: u32 = 0;
            for p in sub_k..k {
                if !prev_sig(p) {
                    zero_count += 1;
                }
            }
            let mut run = zero_count;
            let mut s = sub_k;
            while run >= 16 {
                let mut p = s;
                let mut zeros_in_chunk: u32 = 0;
                while p < k {
                    if !prev_sig(p) {
                        zeros_in_chunk += 1;
                        if zeros_in_chunk == 16 {
                            p += 1;
                            break;
                        }
                    }
                    p += 1;
                }
                debug_assert_eq!(zeros_in_chunk, 16);
                sink.ac_symbol(0xF0)?;
                for r in s..p {
                    if prev_sig(r) {
                        let bit = ((block[r].unsigned_abs() >> al) & 1) as u32;
                        sink.raw_bits(bit, 1)?;
                    }
                }
                s = p;
                run -= 16;
            }
            let symbol = ((run as u8) << 4) | 1;
            sink.ac_symbol(symbol)?;
            let sign = if block[k] > 0 { 1u32 } else { 0 };
            sink.raw_bits(sign, 1)?;
            for r in s..k {
                if prev_sig(r) {
                    let bit = ((block[r].unsigned_abs() >> al) & 1) as u32;
                    sink.raw_bits(bit, 1)?;
                }
            }
            sub_k = k + 1;
        }
        k += 1;
    }

    // Trailing tail (sub_k..=se): only existing-sigs and zeros (no
    // further new-sigs by construction). The decoder is still inside
    // this block expecting a terminator → contribute one EOB to the
    // run, and queue the trailing correction bits for emission after
    // the next flush. The decoder's `refine_existing_band(br, coef,
    // k=sub_k, se, ...)` will consume exactly those bits.
    if sub_k <= se {
        state.extend();
        for r in sub_k..=se {
            if prev_sig(r) {
                let bit = ((block[r].unsigned_abs() >> al) & 1) as u32;
                state.push_ref_bit(bit);
            }
        }
    }

    Ok(())
}

/// Standard-tables (no EOBn) AC-refine per-block emitter. Mirrors the
/// 0.8.0 behavior exactly so the byte stream is unchanged when
/// `allow_eobn = false`.
#[allow(clippy::needless_range_loop)]
fn encode_ac_refine_block_eob0<W: Write>(
    bw: &mut BitWriter<W>,
    block: &[i16; 64],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
    ac_tab: &HuffmanTable,
) -> io::Result<()> {
    let ss = ss as usize;
    let se = se as usize;
    let prev_threshold = 1u16 << ah;
    let prev_sig = |k: usize| block[k].unsigned_abs() >= prev_threshold;
    let new_sig = |k: usize| {
        let av = block[k].unsigned_abs();
        av < prev_threshold && (av >> al) != 0
    };

    let has_new = (ss..=se).any(new_sig);

    if !has_new {
        emit_eob0(bw, ac_tab)?;
        for k in ss..=se {
            if prev_sig(k) {
                let bit = (block[k].unsigned_abs() >> al) & 1;
                bw.write_bits(bit as u32, 1)?;
            }
        }
        return Ok(());
    }

    let mut sub_k = ss;
    let mut k = ss;
    while k <= se {
        if new_sig(k) {
            let mut zero_count: u32 = 0;
            for p in sub_k..k {
                if !prev_sig(p) {
                    zero_count += 1;
                }
            }
            let mut run = zero_count;
            let mut s = sub_k;
            while run >= 16 {
                let mut p = s;
                let mut zeros_in_chunk: u32 = 0;
                while p < k {
                    if !prev_sig(p) {
                        zeros_in_chunk += 1;
                        if zeros_in_chunk == 16 {
                            p += 1;
                            break;
                        }
                    }
                    p += 1;
                }
                debug_assert_eq!(zeros_in_chunk, 16);
                let zrl = ac_tab.packed[0xF0];
                bw.write_bits(zrl & 0xFFFF, zrl >> 16)?;
                for r in s..p {
                    if prev_sig(r) {
                        let bit = (block[r].unsigned_abs() >> al) & 1;
                        bw.write_bits(bit as u32, 1)?;
                    }
                }
                s = p;
                run -= 16;
            }
            let symbol = ((run as usize) << 4) | 1;
            let entry = ac_tab.packed[symbol];
            let code = entry & 0xFFFF;
            let len = entry >> 16;
            bw.write_bits(code, len)?;
            let sign = if block[k] > 0 { 1u32 } else { 0 };
            bw.write_bits(sign, 1)?;
            for r in s..k {
                if prev_sig(r) {
                    let bit = (block[r].unsigned_abs() >> al) & 1;
                    bw.write_bits(bit as u32, 1)?;
                }
            }
            sub_k = k + 1;
        }
        k += 1;
    }

    if sub_k <= se {
        emit_eob0(bw, ac_tab)?;
        for r in sub_k..=se {
            if prev_sig(r) {
                let bit = (block[r].unsigned_abs() >> al) & 1;
                bw.write_bits(bit as u32, 1)?;
            }
        }
    }

    Ok(())
}

// ============================================================================
// JpegEncoder accessor shims
// ============================================================================

impl<W: Write> JpegEncoder<W> {
    pub(crate) fn out_mut(&mut self) -> &mut W {
        &mut self.out
    }
    pub(crate) fn quality(&self) -> u8 {
        self.quality
    }
    pub(crate) fn subsampling(&self) -> ChromaSubsampling {
        self.subsampling
    }
    pub(crate) fn custom_quant(&self) -> Option<(&[u8; 64], &[u8; 64])> {
        self.custom_quant.as_ref().map(|(l, c)| (&**l, &**c))
    }
    pub(crate) fn optimize(&self) -> bool {
        self.optimize_huffman
    }
    pub(crate) fn exif_bytes(&self) -> Option<&[u8]> {
        self.exif.as_deref()
    }
    pub(crate) fn icc_bytes(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AC-first toward-zero shift matches Rust's signed `/`:
    /// positive coefs floor toward zero, negative coefs ceiling
    /// toward zero (= magnitude truncation). Distinct from
    /// arithmetic right shift, which rounds toward -∞.
    #[test]
    fn ac_first_toward_zero_shift() {
        let cases = [
            (3i16, 1u8, 1i32),
            (4, 1, 2),
            (5, 1, 2),
            (-3, 1, -1),
            (-4, 1, -2),
            (-5, 1, -2),
            (-1, 1, 0),
            (1, 1, 0),
            (0, 1, 0),
        ];
        for (coef, al, expected) in cases {
            let abs = (coef.unsigned_abs() >> al) as i32;
            let v = if coef < 0 { -abs } else { abs };
            assert_eq!(v, expected, "coef={coef}, al={al}");
        }
    }

    /// DC refine bit = `((dc as u16) >> al) & 1`.
    #[test]
    fn dc_refine_bit_uses_i16_bit_pattern() {
        let cases = [
            (-3i16, 0u8, 1u32),
            (-4, 0, 0),
            (3, 0, 1),
            (4, 0, 0),
            (-3, 1, 0),
            (3, 1, 1),
        ];
        for (dc, al, expected) in cases {
            let bit = ((dc as u16) >> al) & 1;
            assert_eq!(bit as u32, expected, "dc={dc}, al={al}");
        }
    }

    /// EOBRUN flush picks the largest N ≤ 14 with `(1 << N) ≤ run`,
    /// emits the (sym, extra) pair, and zeroes the run. Verify the
    /// emitted (sym, extra-bits) sequence for representative runs.
    #[test]
    fn eobrun_flush_picks_max_n() {
        struct Capture {
            log: Vec<(u8, u32, u32)>, // (sym, extra, extra_bits) — extra_bits=0 marks no raw bits.
            last_sym: Option<u8>,
        }
        impl Sink for Capture {
            fn ac_symbol(&mut self, sym: u8) -> io::Result<()> {
                self.last_sym = Some(sym);
                self.log.push((sym, 0, 0));
                Ok(())
            }
            fn raw_bits(&mut self, value: u32, n_bits: u32) -> io::Result<()> {
                let last = self.log.last_mut().unwrap();
                last.1 = value;
                last.2 = n_bits;
                let _ = self.last_sym;
                Ok(())
            }
        }
        let cases = [
            (1u32, vec![(0x00u8, 0u32, 0u32)]),
            (2, vec![(0x10, 0, 1)]),
            (3, vec![(0x10, 1, 1)]),
            (8, vec![(0x30, 0, 3)]),
            (15, vec![(0x30, 7, 3)]),
            (16, vec![(0x40, 0, 4)]),
            (17, vec![(0x40, 1, 4)]),
            (
                // Run of 16384 = 2^14, single emit at N=14, extra=0.
                16384,
                vec![(0xE0, 0, 14)],
            ),
            (
                // Run of 16385 = 2^14 + 1: single emit at N=14, extra=1.
                16385,
                vec![(0xE0, 1, 14)],
            ),
            (
                // Run of 32768 = 2^15: one EOBn-14 covers at most
                // `2^15 - 1 = 32767`, so split = (16384+16383) + 1.
                32768,
                vec![(0xE0, 16383, 14), (0x00, 0, 0)],
            ),
        ];
        for (run, expected) in cases {
            let mut sink = Capture {
                log: Vec::new(),
                last_sym: None,
            };
            let mut state = EobrunState {
                run,
                pending_ref_bits: Vec::new(),
            };
            flush_eobrun(&mut sink, &mut state).unwrap();
            assert_eq!(state.run, 0);
            assert_eq!(sink.log, expected, "run={run}");
        }
    }

    /// Pending correction bits get drained after the EOBn header(s).
    #[test]
    fn eobrun_flush_drains_pending_ref_bits() {
        struct Capture {
            calls: Vec<(&'static str, u32, u32)>,
        }
        impl Sink for Capture {
            fn ac_symbol(&mut self, sym: u8) -> io::Result<()> {
                self.calls.push(("sym", sym as u32, 0));
                Ok(())
            }
            fn raw_bits(&mut self, value: u32, n_bits: u32) -> io::Result<()> {
                self.calls.push(("bits", value, n_bits));
                Ok(())
            }
        }
        let mut sink = Capture { calls: Vec::new() };
        let mut state = EobrunState {
            run: 3,
            pending_ref_bits: vec![(1, 1), (0, 1), (1, 1)],
        };
        flush_eobrun(&mut sink, &mut state).unwrap();
        // run=3 → N=1, sym=0x10, extra=1 (1 bit).
        assert_eq!(
            sink.calls,
            vec![
                ("sym", 0x10, 0),
                ("bits", 1, 1),
                ("bits", 1, 1),
                ("bits", 0, 1),
                ("bits", 1, 1),
            ]
        );
        assert!(state.pending_ref_bits.is_empty());
    }
}
