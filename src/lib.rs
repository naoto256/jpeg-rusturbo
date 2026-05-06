//! Baseline JPEG encoder — integer LL&M DCT + per-arch SIMD kernels.
//!
//! Hot kernels live behind a single `arch::backend` dispatch chosen at
//! compile time (`aarch64 + !force-scalar` → NEON; everything else →
//! scalar reference). All SIMD backends produce bit-exact-equivalent
//! output to the scalar reference; cross-check tests assert this in
//! `arch::neon`'s test module.
//!
//! Pipeline:
//!
//! ```text
//!   RGB(A) bytes
//!       │ color::extract_*                    (orchestration)
//!       │   └─ arch::backend::color::rgb_row_to_ycc
//!       ▼
//!   8x8 i16 blocks (level-shifted)
//!       │ arch::backend::dct::fdct_islow      (12-mul integer LL&M DCT)
//!       ▼
//!   8x8 i16 DCT coefficients (scaled by 8)
//!       │ quant::quantize_and_zigzag
//!       │   └─ arch::backend::quant::quantize_natural + scalar zig-zag
//!       ▼
//!   8x8 i16 zig-zag coefficients
//!       │ huffman::encode_block               (Phase 2.5: u64 accumulator,
//!       │   └─ arch::backend::huffman::group_of_8_is_zero (8-skip)
//!       ▼
//!   entropy-coded bytes (with 0xFF → 0xFF 0x00 stuffing)
//! ```
//!
//! See `LICENSES/` for attribution; the SIMD kernels are translations
//! of libjpeg-turbo (BSD-3-Clause + IJG).

mod arch;
pub mod color;
mod huffman;
mod markers;
mod quant;
mod tables;

use std::io::{self, Write};

use crate::color::{PixelLayout, RGB, RGBA};
use crate::huffman::{BitWriter, HuffmanTable, encode_block};
use crate::quant::Divisors;
use crate::tables::{
    STD_CHROMA_AC, STD_CHROMA_DC, STD_CHROMA_QUANT, STD_LUMA_AC, STD_LUMA_DC, STD_LUMA_QUANT,
    scale_quant_table,
};

/// Chroma subsampling mode. We support 4:4:4 (no subsample) and 4:2:0
/// (2x2 chroma block per luma block — what every conventional baseline
/// JPEG ships).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// Y, Cb, Cr at full resolution.
    Yuv444,
    /// Y at full resolution, Cb/Cr at half resolution in both axes.
    Yuv420,
}

/// Drop-in JPEG encoder. Mirrors the `image::codecs::jpeg::JpegEncoder`
/// shape so call sites can be ported with a `use` swap. Default
/// subsampling is 4:2:0 to match what we ship today.
pub struct JpegEncoder<W: Write> {
    out: W,
    quality: u8,
    subsampling: ChromaSubsampling,
}

impl<W: Write> JpegEncoder<W> {
    /// Create an encoder at the given quality (1..=100, clamped). The
    /// inner writer receives the full JPEG byte stream when `encode_*`
    /// is called.
    pub fn new_with_quality(out: W, quality: u8) -> Self {
        Self {
            out,
            quality: quality.clamp(1, 100),
            subsampling: ChromaSubsampling::Yuv420,
        }
    }

    /// Override subsampling. Call before `encode_*`.
    pub fn set_subsampling(&mut self, s: ChromaSubsampling) {
        self.subsampling = s;
    }

    /// Encode an RGB pixel buffer. `pixels.len()` must be at least
    /// `width * height * 3`; trailing bytes are ignored.
    pub fn encode_rgb(&mut self, pixels: &[u8], width: u32, height: u32) -> io::Result<()> {
        self.encode_inner(pixels, width, height, RGB)
    }

    /// Encode an RGBA pixel buffer. The alpha channel is dropped (JPEG
    /// has no alpha). Saves one full-frame copy compared to the
    /// `image` crate's encoder, which only accepts RGB and forces the
    /// caller to repack.
    pub fn encode_rgba(&mut self, pixels: &[u8], width: u32, height: u32) -> io::Result<()> {
        self.encode_inner(pixels, width, height, RGBA)
    }

    fn encode_inner(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> io::Result<()> {
        if width == 0 || height == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "image dimensions must be non-zero",
            ));
        }
        let needed = (width as usize) * (height as usize) * layout.bpp;
        if pixels.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "pixel buffer too small: {} < {} ({}x{}*{})",
                    pixels.len(),
                    needed,
                    width,
                    height,
                    layout.bpp
                ),
            ));
        }

        // Quant tables (8-bit, scaled by quality). Index 0 = luma,
        // 1 = chroma.
        let luma_q = scale_quant_table(&STD_LUMA_QUANT, self.quality);
        let chroma_q = scale_quant_table(&STD_CHROMA_QUANT, self.quality);
        let div_luma = quant::build_divisors(&luma_q);
        let div_chroma = quant::build_divisors(&chroma_q);

        // Expand the standard Huffman tables into encoder lookups.
        let dc_luma = HuffmanTable::from_std(&STD_LUMA_DC);
        let ac_luma = HuffmanTable::from_std(&STD_LUMA_AC);
        let dc_chroma = HuffmanTable::from_std(&STD_CHROMA_DC);
        let ac_chroma = HuffmanTable::from_std(&STD_CHROMA_AC);

        // ---- Header ----
        markers::write_soi(&mut self.out)?;
        markers::write_app0_jfif(&mut self.out)?;
        markers::write_dqt(&mut self.out, 0, &luma_q)?;
        markers::write_dqt(&mut self.out, 1, &chroma_q)?;

        let (h_y, v_y) = match self.subsampling {
            ChromaSubsampling::Yuv444 => (1, 1),
            ChromaSubsampling::Yuv420 => (2, 2),
        };
        markers::write_sof0(
            &mut self.out,
            width as u16,
            height as u16,
            &[
                (1, h_y, v_y, 0), // Y
                (2, 1, 1, 1),     // Cb
                (3, 1, 1, 1),     // Cr
            ],
        )?;

        markers::write_dht(&mut self.out, 0, 0, &STD_LUMA_DC)?;
        markers::write_dht(&mut self.out, 1, 0, &STD_LUMA_AC)?;
        markers::write_dht(&mut self.out, 0, 1, &STD_CHROMA_DC)?;
        markers::write_dht(&mut self.out, 1, 1, &STD_CHROMA_AC)?;

        markers::write_sos(
            &mut self.out,
            &[
                (1, 0, 0), // Y → DC0/AC0
                (2, 1, 1), // Cb → DC1/AC1
                (3, 1, 1), // Cr → DC1/AC1
            ],
        )?;

        // ---- Entropy-coded segment ----
        let mut bw = BitWriter::new(&mut self.out);
        // Crude upper bound: ~1 byte per pixel for q=80 typical content;
        // the bitwriter resizes as needed but starting close avoids the
        // first few reallocations on large frames.
        bw.reserve((width as usize) * (height as usize));
        match self.subsampling {
            ChromaSubsampling::Yuv444 => encode_scan_444(
                pixels, width, height, layout,
                &mut bw, &div_luma, &div_chroma,
                &dc_luma, &ac_luma, &dc_chroma, &ac_chroma,
            )?,
            ChromaSubsampling::Yuv420 => encode_scan_420(
                pixels, width, height, layout,
                &mut bw, &div_luma, &div_chroma,
                &dc_luma, &ac_luma, &dc_chroma, &ac_chroma,
            )?,
        }
        bw.flush_to_byte_boundary()?;

        // ---- Trailer ----
        markers::write_eoi(&mut self.out)?;
        Ok(())
    }

}

#[allow(clippy::too_many_arguments)]
fn encode_scan_444<BW: Write>(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    bw: &mut BitWriter<BW>,
    div_luma: &Divisors,
    div_chroma: &Divisors,
    dc_luma: &HuffmanTable,
    ac_luma: &HuffmanTable,
    dc_chroma: &HuffmanTable,
    ac_chroma: &HuffmanTable,
) -> io::Result<()> {
    let mcus_x = width.div_ceil(8);
    let mcus_y = height.div_ceil(8);
    let mut prev_dc_y = 0i16;
    let mut prev_dc_cb = 0i16;
    let mut prev_dc_cr = 0i16;
    let mut y_blk = [0i16; 64];
    let mut cb_blk = [0i16; 64];
    let mut cr_blk = [0i16; 64];

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            color::extract_block_ycbcr(
                pixels, width, height, layout,
                mx * 8, my * 8,
                &mut y_blk, &mut cb_blk, &mut cr_blk,
            );

            prev_dc_y = encode_one_block(bw, &mut y_blk, div_luma, prev_dc_y, dc_luma, ac_luma)?;
            prev_dc_cb = encode_one_block(bw, &mut cb_blk, div_chroma, prev_dc_cb, dc_chroma, ac_chroma)?;
            prev_dc_cr = encode_one_block(bw, &mut cr_blk, div_chroma, prev_dc_cr, dc_chroma, ac_chroma)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_scan_420<BW: Write>(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    bw: &mut BitWriter<BW>,
    div_luma: &Divisors,
    div_chroma: &Divisors,
    dc_luma: &HuffmanTable,
    ac_luma: &HuffmanTable,
    dc_chroma: &HuffmanTable,
    ac_chroma: &HuffmanTable,
) -> io::Result<()> {
    let mcus_x = width.div_ceil(16);
    let mcus_y = height.div_ceil(16);
    let mut prev_dc_y = 0i16;
    let mut prev_dc_cb = 0i16;
    let mut prev_dc_cr = 0i16;
    let mut y_blocks = [[0i16; 64]; 4];
    let mut cb_blk = [0i16; 64];
    let mut cr_blk = [0i16; 64];

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            color::extract_mcu_420(
                pixels, width, height, layout,
                mx * 16, my * 16,
                &mut y_blocks, &mut cb_blk, &mut cr_blk,
            );

            for blk in y_blocks.iter_mut() {
                prev_dc_y = encode_one_block(bw, blk, div_luma, prev_dc_y, dc_luma, ac_luma)?;
            }
            prev_dc_cb = encode_one_block(bw, &mut cb_blk, div_chroma, prev_dc_cb, dc_chroma, ac_chroma)?;
            prev_dc_cr = encode_one_block(bw, &mut cr_blk, div_chroma, prev_dc_cr, dc_chroma, ac_chroma)?;
        }
    }
    Ok(())
}

/// Run one block end-to-end: DCT → quantize+zigzag → entropy-code.
fn encode_one_block<W: Write>(
    bw: &mut BitWriter<W>,
    block: &mut [i16; 64],
    div: &Divisors,
    prev_dc: i16,
    dc_tab: &HuffmanTable,
    ac_tab: &HuffmanTable,
) -> io::Result<i16> {
    arch::backend::dct::fdct_islow(block);
    let zz = quant::quantize_and_zigzag(block, div);
    encode_block(bw, &zz, prev_dc, dc_tab, ac_tab)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: SOI + EOI bracket a non-empty payload.
    #[test]
    fn produces_soi_eoi_markers() {
        let w = 16;
        let h = 16;
        let mut rgba = Vec::with_capacity(w * h * 4);
        for _ in 0..(w * h) {
            rgba.extend_from_slice(&[200, 100, 50, 255]);
        }
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.encode_rgba(&rgba, w as u32, h as u32).unwrap();
        assert_eq!(&out[..2], &[0xFF, 0xD8], "missing SOI");
        assert_eq!(&out[out.len() - 2..], &[0xFF, 0xD9], "missing EOI");
        assert!(out.len() > 100, "stream too short to be plausible");
    }

    #[test]
    fn rejects_short_buffer() {
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        let pixels = vec![0u8; 4 * 4 * 2]; // half what 4x4 RGBA needs
        let err = enc.encode_rgba(&pixels, 4, 4).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn rejects_zero_dimensions() {
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        let err = enc.encode_rgb(&[], 0, 8).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
