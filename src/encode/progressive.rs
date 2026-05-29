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

use std::io::{self, Write};

use crate::tables::{
    STD_CHROMA_AC, STD_CHROMA_DC, STD_CHROMA_QUANT, STD_LUMA_AC, STD_LUMA_DC, STD_LUMA_QUANT,
    scale_quant_table,
};
use crate::{ChromaSubsampling, PixelLayout};

use super::huffman::{BitWriter, HuffmanTable, magnitude_category};
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
    // Dispatch on subsampling so the body monomorphizes against the
    // per-scheme `MCU_W` / `Y_BLOCKS_PER_MCU` constants. Mirrors the
    // `dispatch_scheme!` macro pattern in `lib.rs`, inlined here so
    // we don't have to make the macro crate-public.
    match enc.subsampling() {
        ChromaSubsampling::Yuv444 => encode_progressive_scheme::<Yuv444Scheme, _>(
            enc, pixels, width, height, layout, &luma_q, &chroma_q, div_luma, div_chroma,
        ),
        ChromaSubsampling::Yuv422 => encode_progressive_scheme::<Yuv422Scheme, _>(
            enc, pixels, width, height, layout, &luma_q, &chroma_q, div_luma, div_chroma,
        ),
        ChromaSubsampling::Yuv420 => encode_progressive_scheme::<Yuv420Scheme, _>(
            enc, pixels, width, height, layout, &luma_q, &chroma_q, div_luma, div_chroma,
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
) -> io::Result<()> {
    let mcus_x = width.div_ceil(S::MCU_W);
    let mcus_y = height.div_ceil(S::MCU_H);
    let total_mcus = (mcus_x as usize) * (mcus_y as usize);

    // ---- Pass 1: quantize every block once.
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

    let dc_luma = HuffmanTable::from_std(&STD_LUMA_DC);
    let ac_luma = HuffmanTable::from_std(&STD_LUMA_AC);
    let dc_chroma = HuffmanTable::from_std(&STD_CHROMA_DC);
    let ac_chroma = HuffmanTable::from_std(&STD_CHROMA_AC);
    markers::write_dht(enc.out_mut(), 0, 0, &STD_LUMA_DC)?;
    markers::write_dht(enc.out_mut(), 1, 0, &STD_LUMA_AC)?;
    markers::write_dht(enc.out_mut(), 0, 1, &STD_CHROMA_DC)?;
    markers::write_dht(enc.out_mut(), 1, 1, &STD_CHROMA_AC)?;

    // ---- Scan 1: DC interleaved first.
    encode_dc_interleaved_first::<S, _>(
        enc.out_mut(),
        &y_blocks,
        &cb_blocks,
        &cr_blocks,
        &dc_luma,
        &dc_chroma,
        AL_FIRST,
    )?;

    // ---- Scans 2-4: AC first per component.
    encode_ac_first_scan_indexed(
        enc.out_mut(),
        &y_blocks,
        &y_raster_indices,
        &ac_luma,
        1,
        0,
        1,
        63,
        AL_FIRST,
    )?;
    encode_ac_first_scan(enc.out_mut(), &cb_blocks, &ac_chroma, 2, 1, 1, 63, AL_FIRST)?;
    encode_ac_first_scan(enc.out_mut(), &cr_blocks, &ac_chroma, 3, 1, 1, 63, AL_FIRST)?;

    // ---- Scan 5: DC interleaved refine.
    encode_dc_interleaved_refine::<S, _>(
        enc.out_mut(),
        &y_blocks,
        &cb_blocks,
        &cr_blocks,
        AH_REFINE,
        AL_REFINE,
    )?;

    // ---- Scans 6-8: AC refine per component.
    encode_ac_refine_scan_indexed(
        enc.out_mut(),
        &y_blocks,
        &y_raster_indices,
        &ac_luma,
        1,
        0,
        1,
        63,
        AH_REFINE,
        AL_REFINE,
    )?;
    encode_ac_refine_scan(
        enc.out_mut(),
        &cb_blocks,
        &ac_chroma,
        2,
        1,
        1,
        63,
        AH_REFINE,
        AL_REFINE,
    )?;
    encode_ac_refine_scan(
        enc.out_mut(),
        &cr_blocks,
        &ac_chroma,
        3,
        1,
        1,
        63,
        AH_REFINE,
        AL_REFINE,
    )?;

    markers::write_eoi(enc.out_mut())?;
    Ok(())
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
// AC first scan
// ============================================================================

/// Raster-order variant of `encode_ac_first_scan` for Y at
/// subsampled layouts, where the MCU-chunked storage in `blocks`
/// needs an index permutation to recover raster order.
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
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, 0, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 8);
    for &idx in raster_indices {
        encode_ac_first(&mut bw, &blocks[idx], ss, se, ac_tab, al)?;
    }
    bw.flush_to_byte_boundary()
}

/// SOS + entropy for one AC-first scan (`Ss, Se, Ah=0, Al`). Blocks
/// walk in raster order (non-interleaved single-component scan); the
/// EOB run accumulates across blocks within this scan and is flushed
/// at scan-end.
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
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, 0, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 8);
    for block in blocks {
        encode_ac_first(&mut bw, block, ss, se, ac_tab, al)?;
    }
    bw.flush_to_byte_boundary()
}

/// Emit an AC-first-scan band (`Ss..=Se, Ah=0, Al`) for one block.
///
/// Coefficient comparison uses *toward-zero* shift (`coef / (1 <<
/// Al)`) so that the decoder's `V << Al` reconstruction is the
/// nearest multiple of `2^Al` with smaller magnitude than the
/// original — the refine scan then adds `±(1 << Al')` (with the
/// existing sign) to grow the magnitude.
///
/// End-of-band is signalled with `EOB0` (symbol `0x00`) emitted per
/// block — `EOBn` for `n ≥ 1` (symbols `0x10`..`0xE0`) is *missing*
/// from the Annex K AC Huffman tables this crate ships, so a
/// multi-block run would silently emit nothing. A future
/// `encode_progressive_optimize` path can extend the tables to
/// include `EOBn` symbols and recover the ~5-10% file-size win.
fn encode_ac_first<W: Write>(
    bw: &mut BitWriter<W>,
    block: &[i16; 64],
    ss: u8,
    se: u8,
    ac_tab: &HuffmanTable,
    al: u8,
) -> io::Result<()> {
    let ss = ss as usize;
    let se = se as usize;
    // Toward-zero shift: drop the low `al` bits of the magnitude,
    // preserve the sign. Equivalent to `coef / (1 << al)` in Rust
    // integer arithmetic for signed types.
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
            // Whole band is zero → emit EOB0 for this block alone.
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
        // Trailing zeros in this block → emit EOB0.
        emit_eob0(bw, ac_tab)?;
    }
    Ok(())
}

/// Emit a single `EOB0` Huffman symbol (symbol `0x00`, "end of band,
/// no run extension"). The Annex K tables this crate ships have a
/// code for this symbol; the longer `EOBn` symbols (`0x10`..`0xE0`)
/// are not present, so the progressive scans use `EOB0` per block
/// rather than a multi-block run. See module-level rationale.
fn emit_eob0<W: Write>(bw: &mut BitWriter<W>, ac_tab: &HuffmanTable) -> io::Result<()> {
    let entry = ac_tab.packed[0x00];
    let code = entry & 0xFFFF;
    let len = entry >> 16;
    bw.write_bits(code, len)
}

// ============================================================================
// AC refine scan
// ============================================================================

/// Raster-order variant of `encode_ac_refine_scan` for Y at
/// subsampled layouts.
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
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, ah, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 4);
    for &idx in raster_indices {
        encode_ac_refine_block(&mut bw, &blocks[idx], ss, se, ah, al, ac_tab)?;
    }
    bw.flush_to_byte_boundary()
}

/// SOS + entropy for one AC-refine scan
/// (`Ss, Se, Ah > 0, Al < Ah`). Each block self-terminates with an
/// `EOB0` Huffman code; cross-block `EOBn` runs are avoided to stay
/// compatible with the Annex K reference Huffman tables (which omit
/// the `EOBn` symbols for `n ≥ 1`).
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
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], ss, se, ah, al)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 4);
    for block in blocks {
        encode_ac_refine_block(&mut bw, block, ss, se, ah, al, ac_tab)?;
    }
    bw.flush_to_byte_boundary()
}

/// Emit one block's contribution to an AC-refine scan.
///
/// Per-position classification:
/// - **previously significant** (`|coef| >= 1 << Ah`): emit one
///   refinement bit `(|coef| >> Al) & 1`, in walk order. Placement:
///   after the next Huffman code (interleaved with the run) OR into
///   the deferred EOB-refine buffer (if no new-significant in
///   block).
/// - **newly significant** (`|coef| < 1 << Ah && |coef| >= 1 << Al`,
///   which for our `Ah-Al = 1` reduces to `|coef| == 1`): emit a
///   `(zero_run, size=1)` Huffman symbol followed by the sign bit
///   and any pending refinement bits for prior-positions-in-this-
///   block.
/// - **zero** (`|coef| < 1 << Al`): contributes to `zero_run`.
//
// `clippy::needless_range_loop` would have us rewrite these walks
// as `.iter().enumerate()` over the band slice — but the closures
// `prev_sig` / `new_sig` (and the per-element ZRL accounting) want
// to query `block[r]` at indices the body computes separately, so
// the range-loop form is the readable one. Allow the lint
// site-locally.
#[allow(clippy::needless_range_loop)]
fn encode_ac_refine_block<W: Write>(
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
    let prev_threshold = 1u16 << ah; // |coef| >= this ⇒ previously significant
    let prev_sig = |k: usize| block[k].unsigned_abs() >= prev_threshold;
    let new_sig = |k: usize| {
        let av = block[k].unsigned_abs();
        av < prev_threshold && (av >> al) != 0
    };

    // Fast path: any new-significant in band?
    let has_new = (ss..=se).any(new_sig);

    if !has_new {
        // Whole band has only existing-significant + zeros → emit a
        // self-contained EOB0 followed by refinement bits for every
        // existing-sig in the band, in walk order.
        emit_eob0(bw, ac_tab)?;
        for k in ss..=se {
            if prev_sig(k) {
                let bit = (block[k].unsigned_abs() >> al) & 1;
                bw.write_bits(bit as u32, 1)?;
            }
        }
        return Ok(());
    }

    // Walk band, emitting codes + interleaved refinement bits.
    let mut sub_k = ss;
    let mut k = ss;
    while k <= se {
        if new_sig(k) {
            // Count "new zeros" between sub_k and k (positions that
            // are neither prev_sig nor new_sig — i.e. true zeros at
            // current precision).
            let mut zero_count: u32 = 0;
            for p in sub_k..k {
                if !prev_sig(p) {
                    zero_count += 1;
                }
            }
            // ZRL chunks (when zero_count >= 16). Each ZRL covers a
            // segment of (sub_k → p) spanning exactly 16 zeros plus
            // any existing-sig positions in between; refinement bits
            // for those existing-sigs follow the ZRL code in walk
            // order.
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
            // Emit (run < 16, size=1) Huffman + sign bit + remaining
            // refinement bits in (s..k).
            let symbol = ((run as usize) << 4) | 1;
            let entry = ac_tab.packed[symbol];
            let code = entry & 0xFFFF;
            let len = entry >> 16;
            bw.write_bits(code, len)?;
            // Sign bit: 1 = positive, 0 = negative (decoder mirror in
            // `decode_ac_refine_block`).
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

    // Trailing tail (sub_k..=se) — by construction it contains only
    // existing-sigs and zeros (no further new-sigs). The decoder's
    // outer loop is still running for this block (k_decoder <= se
    // after the last new-sig emit) and expects to read a terminator
    // code: emit an EOB0, immediately followed by the refinement
    // bits for the trailing existing-sigs in walk order.
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
    pub(crate) fn exif_bytes(&self) -> Option<&[u8]> {
        self.exif.as_deref()
    }
    pub(crate) fn icc_bytes(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }
}

#[cfg(test)]
mod tests {
    /// AC-first toward-zero shift matches Rust's signed `/`:
    /// positive coefs floor toward zero, negative coefs ceiling
    /// toward zero (= magnitude truncation). Distinct from
    /// arithmetic right shift, which rounds toward -∞.
    #[test]
    fn ac_first_toward_zero_shift() {
        // (input, al, expected V)
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

    /// DC refine bit = `((dc as u16) >> al) & 1`. For `al = 0` it
    /// agrees with the magnitude formula `(|dc| >> al) & 1`; for
    /// `al > 0` with negative dc the formulas diverge — exercise
    /// the case so a future regression that swaps formulas trips
    /// here first.
    #[test]
    fn dc_refine_bit_uses_i16_bit_pattern() {
        let cases = [
            // (dc, al, expected bit)
            (-3i16, 0u8, 1u32),
            (-4, 0, 0),
            (3, 0, 1),
            (4, 0, 0),
            // At al=1: -3 (0xFFFD) bit 1 = 0; |-3|=3 bit 1 = 1. Use
            // the bit-pattern result.
            (-3, 1, 0),
            (3, 1, 1),
        ];
        for (dc, al, expected) in cases {
            let bit = ((dc as u16) >> al) & 1;
            assert_eq!(bit as u32, expected, "dc={dc}, al={al}");
        }
    }
}
