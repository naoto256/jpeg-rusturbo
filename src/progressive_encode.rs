//! Progressive (SOF2) JPEG encoder — foundation.
//!
//! Splits the entropy-coded segment into multiple scans, each emitting
//! a sub-band of every block's coefficients (T.81 Annex G). This file
//! ships the **spectral** half of the progressive grammar:
//!
//! - DC first scans (`Ss=0, Se=0, Ah=0, Al≥0`): emit `DC >> Al` with
//!   the standard magnitude-category encoding.
//! - AC first scans (`Ss≥1, Se≤63, Ah=0, Al≥0`): emit `coef >> Al` in
//!   the band, with EOB-run encoding (T.81 G.1.2.2 / G.2) folding
//!   trailing-zero blocks into a single suffix.
//!
//! Successive approximation refinement scans (`Ah > 0`) — the bit-
//! slice + EOB-run-with-refinement layer — live in a follow-up; the
//! decoder side already supports the full grammar so the encoder can
//! extend without touching this module's contracts.
//!
//! The first cut ships a four-scan spectral plan at `Al=0`:
//!
//! 1. **DC interleaved** — `Ss=0, Se=0`, all three components in one
//!    MCU stream.
//! 2. **AC Y full band** — `Ss=1, Se=63`, Y only.
//! 3. **AC Cb full band** — same shape, Cb.
//! 4. **AC Cr full band** — same shape, Cr.
//!
//! Output is a valid progressive JPEG decodable by every conforming
//! progressive decoder (including the one in this crate).

use std::io::{self, Write};

use crate::huffman::{BitWriter, HuffmanTable, magnitude_category};
use crate::markers;
use crate::quant::Divisors;
use crate::tables::{
    STD_CHROMA_AC, STD_CHROMA_DC, STD_CHROMA_QUANT, STD_LUMA_AC, STD_LUMA_DC, STD_LUMA_QUANT,
    scale_quant_table,
};
use crate::{
    ChromaSubsampling, DcPredictors, JpegEncoder, PixelLayout, SamplingScheme, Yuv420Scheme,
    Yuv422Scheme, Yuv444Scheme,
};

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
    // Progressive emits the *same* coefficients across multiple scans
    // (DC then AC, per-component), so we materialize all blocks
    // up front and let each scan read from the buffer. Storage cost
    // is `≈ 2 bytes/pixel × subsampling factor`; a 4K frame fits in
    // ~13 MB.
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

    // ---- Header emission.
    //
    // Metadata segments (APP1 EXIF / APP2 ICC) are emitted as raw
    // copies — taking a snapshot via `to_vec` releases the immutable
    // borrow on `enc` before we reach for `enc.out_mut()`.
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
    // SOF2 distinguishes "progressive" from "baseline" — same
    // component layout, different marker byte.
    markers::write_sof2(
        enc.out_mut(),
        width as u16,
        height as u16,
        &[(1, h_y, v_y, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
    )?;

    // Standard canonical Huffman tables. Per-image optimized Huffman
    // for progressive is a 0.9.0 enhancement.
    let dc_luma = HuffmanTable::from_std(&STD_LUMA_DC);
    let ac_luma = HuffmanTable::from_std(&STD_LUMA_AC);
    let dc_chroma = HuffmanTable::from_std(&STD_CHROMA_DC);
    let ac_chroma = HuffmanTable::from_std(&STD_CHROMA_AC);
    markers::write_dht(enc.out_mut(), 0, 0, &STD_LUMA_DC)?;
    markers::write_dht(enc.out_mut(), 1, 0, &STD_LUMA_AC)?;
    markers::write_dht(enc.out_mut(), 0, 1, &STD_CHROMA_DC)?;
    markers::write_dht(enc.out_mut(), 1, 1, &STD_CHROMA_AC)?;

    // ---- Scan 1: DC interleaved (Y / Cb / Cr).
    encode_dc_interleaved_scan::<S, _>(
        enc.out_mut(),
        &y_blocks,
        &cb_blocks,
        &cr_blocks,
        &dc_luma,
        &dc_chroma,
    )?;

    // ---- Scans 2-4: per-component AC full band.
    encode_ac_scan(enc.out_mut(), &y_blocks, &ac_luma, 1, 0)?;
    encode_ac_scan(enc.out_mut(), &cb_blocks, &ac_chroma, 2, 1)?;
    encode_ac_scan(enc.out_mut(), &cr_blocks, &ac_chroma, 3, 1)?;

    markers::write_eoi(enc.out_mut())?;
    Ok(())
}

/// SOS + entropy for the interleaved DC scan (`Ss=0, Se=0, Ah=0,
/// Al=0`). All three components walk in lockstep using the same MCU
/// ordering the baseline path emits.
fn encode_dc_interleaved_scan<S: SamplingScheme, W: Write>(
    out: &mut W,
    y_blocks: &[[i16; 64]],
    cb_blocks: &[[i16; 64]],
    cr_blocks: &[[i16; 64]],
    dc_luma: &HuffmanTable,
    dc_chroma: &HuffmanTable,
) -> io::Result<()> {
    markers::write_sos_spectral(
        out,
        &[(1, 0, 0), (2, 1, 0), (3, 1, 0)],
        0,
        0,
        0,
        0,
    )?;
    let mut bw = BitWriter::new(out);
    bw.reserve(y_blocks.len() * 4);
    let mut prev_dc = DcPredictors::default();
    let y_mcus = y_blocks.chunks_exact(S::Y_BLOCKS_PER_MCU);
    for (y_chunk, (cb, cr)) in y_mcus.zip(cb_blocks.iter().zip(cr_blocks.iter())) {
        for y in y_chunk {
            prev_dc.y = encode_dc_first(&mut bw, y[0], prev_dc.y, dc_luma)?;
        }
        prev_dc.cb = encode_dc_first(&mut bw, cb[0], prev_dc.cb, dc_chroma)?;
        prev_dc.cr = encode_dc_first(&mut bw, cr[0], prev_dc.cr, dc_chroma)?;
    }
    bw.flush_to_byte_boundary()
}

/// SOS + entropy for one AC scan (`Ss=1, Se=63, Ah=0, Al=0`). Blocks
/// are walked in raster order (non-interleaved single-component
/// scan). EOB-run accumulates across blocks within this scan and is
/// flushed at scan-end.
fn encode_ac_scan<W: Write>(
    out: &mut W,
    blocks: &[[i16; 64]],
    ac_tab: &HuffmanTable,
    component_id: u8,
    ac_tab_id: u8,
) -> io::Result<()> {
    markers::write_sos_spectral(out, &[(component_id, 0, ac_tab_id)], 1, 63, 0, 0)?;
    let mut bw = BitWriter::new(out);
    bw.reserve(blocks.len() * 8);
    let mut eobrun: u32 = 0;
    for block in blocks {
        encode_ac_first(&mut bw, block, 1, 63, ac_tab, &mut eobrun)?;
    }
    // End-of-scan: any deferred EOB-run must be flushed before
    // padding to the byte boundary, or the decoder will miss the
    // final blocks' EOB signal.
    flush_eobrun(&mut bw, ac_tab, &mut eobrun)?;
    bw.flush_to_byte_boundary()
}

/// Emit a DC-first-scan token for one block at `Al = 0`. Same math as
/// the baseline DC step minus the AC tail.
fn encode_dc_first<W: Write>(
    bw: &mut BitWriter<W>,
    dc: i16,
    prev_dc: i16,
    dc_tab: &HuffmanTable,
) -> io::Result<i16> {
    let diff = dc as i32 - prev_dc as i32;
    let (size, bits) = magnitude_category(diff);
    let entry = dc_tab.packed[size as usize];
    let code = entry & 0xFFFF;
    let len = entry >> 16;
    // Fused (Huffman code, magnitude bits) write — same trick as
    // `huffman::encode_block`.
    bw.write_bits((code << size) | bits, len + size as u32)?;
    Ok(dc)
}

/// Emit an AC-first-scan band (`Ss..=Se, Al=0`) for one block.
/// Updates `eobrun` and writes any deferred EOB-run flush +
/// non-zero-coefficient symbols required by this block.
///
/// EOB-run accounting:
/// - Block whose `Ss..=Se` band is entirely zero ⇒ `eobrun += 1`.
/// - Block with at least one non-zero in band ⇒ flush deferred
///   `eobrun` first, then emit (zero_run, magnitude) tokens for the
///   non-zeros, trailing zeros in band start a new run-of-1.
fn encode_ac_first<W: Write>(
    bw: &mut BitWriter<W>,
    block: &[i16; 64],
    ss: usize,
    se: usize,
    ac_tab: &HuffmanTable,
    eobrun: &mut u32,
) -> io::Result<()> {
    // Find the last non-zero index inside the band; trailing zeros
    // in `Ss..=Se` fold into the EOB run instead of a per-block EOB.
    let mut last_nz: Option<usize> = None;
    for k in ss..=se {
        if block[k] != 0 {
            last_nz = Some(k);
        }
    }
    let last_nz = match last_nz {
        None => {
            // Whole band zero → grow the EOB run. Flush at the 15-bit
            // ceiling so the next increment doesn't overflow.
            *eobrun += 1;
            if *eobrun == 0x7FFF {
                flush_eobrun(bw, ac_tab, eobrun)?;
            }
            return Ok(());
        }
        Some(k) => k,
    };
    // Non-empty band → any deferred EOB run is emitted *before* this
    // block's symbols.
    flush_eobrun(bw, ac_tab, eobrun)?;

    let mut zero_run: u32 = 0;
    for k in ss..=last_nz {
        let coef = block[k];
        if coef == 0 {
            zero_run += 1;
            continue;
        }
        // ZRL emits for runs ≥ 16; identical encoding to baseline.
        while zero_run >= 16 {
            let zrl = ac_tab.packed[0xF0];
            bw.write_bits(zrl & 0xFFFF, zrl >> 16)?;
            zero_run -= 16;
        }
        let (size, bits) = magnitude_category(coef as i32);
        debug_assert!(size <= 10, "AC magnitude category {size} > 10");
        let symbol = ((zero_run as usize) << 4) | (size as usize & 0x0F);
        let entry = ac_tab.packed[symbol];
        let code = entry & 0xFFFF;
        let len = entry >> 16;
        bw.write_bits((code << size) | bits, len + size as u32)?;
        zero_run = 0;
    }
    // Trailing zeros between `last_nz` and `Se` start a new EOB run
    // for the next block — we don't emit a per-block EOB symbol in
    // progressive AC scans.
    if last_nz < se {
        *eobrun += 1;
        if *eobrun == 0x7FFF {
            flush_eobrun(bw, ac_tab, eobrun)?;
        }
    }
    Ok(())
}

/// Emit the deferred EOB-run as an `EOBn` Huffman symbol + `n` bits
/// of suffix (T.81 G.1.2.2). No-op when `*eobrun == 0`.
///
/// Layout: symbol `(n << 4) | 0x0` where `n` is the bit-length of the
/// run (`0..=14`); the low `n` bits of the run follow as a literal
/// suffix (= `run - 2^n`). `n == 0` covers the `run == 1` case (no
/// suffix).
pub(crate) fn flush_eobrun<W: Write>(
    bw: &mut BitWriter<W>,
    ac_tab: &HuffmanTable,
    eobrun: &mut u32,
) -> io::Result<()> {
    if *eobrun == 0 {
        return Ok(());
    }
    debug_assert!(*eobrun < 0x8000, "eobrun out of range: {}", *eobrun);
    let n = 31 - eobrun.leading_zeros();
    let symbol = (n as usize) << 4;
    let entry = ac_tab.packed[symbol];
    let code = entry & 0xFFFF;
    let len = entry >> 16;
    if n > 0 {
        let suffix = *eobrun & ((1u32 << n) - 1);
        // `len ≤ 16` and `n ≤ 14`, so the combined width is ≤ 30 —
        // mostly within the 27-bit `write_bits` budget but defensive
        // split for the rare table-corruption case where `len = 16`
        // and `n ≥ 12`.
        if len + n <= 27 {
            bw.write_bits((code << n) | suffix, len + n)?;
        } else {
            bw.write_bits(code, len)?;
            bw.write_bits(suffix, n)?;
        }
    } else {
        bw.write_bits(code, len)?;
    }
    *eobrun = 0;
    Ok(())
}

/// Read-only accessor surface for fields the progressive module
/// needs from `JpegEncoder`. Defined as inherent `pub(crate)` methods
/// to keep the encoder's field layout private from this module while
/// still allowing access to what it needs.
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
    /// EOBn bit-length derivation matches T.81 G.1.2.2 (run = 2^n +
    /// suffix). The flush_eobrun path uses `31 - leading_zeros` to
    /// compute `n`; verify on the boundary cases.
    #[test]
    fn eobrun_n_assignment() {
        let cases = [
            (1u32, 0u32),
            (2, 1),
            (3, 1),
            (4, 2),
            (7, 2),
            (8, 3),
            (15, 3),
            (16, 4),
            (1024, 10),
            (16383, 13),
            (32767, 14),
        ];
        for (run, expected_n) in cases {
            let n = 31 - run.leading_zeros();
            assert_eq!(n, expected_n, "run={run}");
        }
    }
}
