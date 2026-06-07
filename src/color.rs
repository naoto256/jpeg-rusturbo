//! YCbCr block extraction with edge replication.
//!
//! The per-row RGB→YCbCr and chroma downsample kernels live in
//! `crate::arch::backend::color`. This file orchestrates them at the
//! 8x8 block / MCU level (4:4:4, 4:2:2, 4:2:0) — padding, level
//! shift, layout distribution.
//!
//! Algorithmic constants for the libjpeg-turbo color transform:
//!   Y  =  0.29900 R + 0.58700 G + 0.11400 B
//!   Cb = -0.16874 R - 0.33126 G + 0.50000 B + 128
//!   Cr =  0.50000 R - 0.41869 G - 0.08131 B + 128
//! all encoded as 16-bit fixed-point and descaled by 16 after summing.

use crate::arch;

pub(crate) const FIX: i32 = 16;
pub(crate) const FIX_HALF: i32 = 1 << (FIX - 1);

pub(crate) const Y_R: u32 = 19595;
pub(crate) const Y_G: u32 = 38470;
pub(crate) const Y_B: u32 = 7471;

pub(crate) const CB_R: u32 = 11059;
pub(crate) const CB_G: u32 = 21709;
pub(crate) const CB_B: u32 = 32768;

pub(crate) const CR_R: u32 = 32768;
pub(crate) const CR_G: u32 = 27439;
pub(crate) const CR_B: u32 = 5329;

// "0.5 + 128 in fixed-point Cb/Cr space" — see libjpeg `jccolext-neon.c`:
// `scaled_128_5 = (128 << 16) + 32767`. The +32767 is a rounding bias
// rolled into the +128 constant; `vshrn_n_u32(x, 16)` is non-rounding,
// so we pre-bias once.
pub(crate) const CHROMA_BIAS: u32 = (128 << 16) + 32767;

/// Per-pixel layout descriptor: bytes per pixel plus the R/G/B byte
/// offsets within a pixel. The alpha / pad byte (when present) lives
/// at the leftover offset and is ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelLayout {
    pub bpp: usize,
    pub r_off: usize,
    pub g_off: usize,
    pub b_off: usize,
    /// True for the 4-byte CMYK pass-through layout. Disambiguates
    /// `bpp == 4 && is_cmyk == true` (raw C/M/Y/K — no RGB↔YCbCr
    /// conversion) from the eight existing 4-byte color layouts
    /// (RGBA / BGRA / ARGB / ABGR / RGBX / BGRX), which all carry RGB
    /// in some byte order. Callers should branch through
    /// [`PixelLayout::class`] rather than reading this field directly.
    pub is_cmyk: bool,
}

/// High-level category of a [`PixelLayout`]. The encode and decode
/// pipelines branch on this single discriminator before any
/// color-kernel dispatch; the per-arch color kernels only ever see
/// [`PixelClass::Rgb`] layouts.
///
/// Derived from the (bpp, is_cmyk) fields of [`PixelLayout`] rather
/// than stored separately, so each pixel-layout constant remains the
/// single source of truth for its category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelClass {
    /// Standard 3- or 4-byte RGB-flavoured layouts: RGB, BGR, RGBA,
    /// BGRA, ARGB, ABGR, RGBX, BGRX. Goes through the YCbCr color
    /// path (with alpha/pad bytes dropped).
    Rgb,
    /// Single-byte luma (the byte **is** Y). Skips color convert and
    /// chroma upsample entirely.
    Gray,
    /// 4-byte CMYK pass-through (C, M, Y, K). No RGB↔YCbCr transform
    /// is performed in either direction.
    Cmyk,
}

impl PixelLayout {
    /// Single discriminator over the three layout categories. Use this
    /// (typically via `match layout.class()`) when the encode or decode
    /// pipeline needs to branch on category — keeps callers from
    /// reading the (bpp, is_cmyk) fields directly and centralises the
    /// "which kernel path do we take" decision in one place.
    #[inline]
    pub fn class(&self) -> PixelClass {
        if self.bpp == 1 {
            PixelClass::Gray
        } else if self.is_cmyk {
            PixelClass::Cmyk
        } else {
            PixelClass::Rgb
        }
    }
}

pub const RGB: PixelLayout = PixelLayout {
    bpp: 3,
    r_off: 0,
    g_off: 1,
    b_off: 2,
    is_cmyk: false,
};
pub const BGR: PixelLayout = PixelLayout {
    bpp: 3,
    r_off: 2,
    g_off: 1,
    b_off: 0,
    is_cmyk: false,
};
pub const RGBA: PixelLayout = PixelLayout {
    bpp: 4,
    r_off: 0,
    g_off: 1,
    b_off: 2,
    is_cmyk: false,
};
pub const BGRA: PixelLayout = PixelLayout {
    bpp: 4,
    r_off: 2,
    g_off: 1,
    b_off: 0,
    is_cmyk: false,
};
pub const ARGB: PixelLayout = PixelLayout {
    bpp: 4,
    r_off: 1,
    g_off: 2,
    b_off: 3,
    is_cmyk: false,
};
pub const ABGR: PixelLayout = PixelLayout {
    bpp: 4,
    r_off: 3,
    g_off: 2,
    b_off: 1,
    is_cmyk: false,
};
pub const RGBX: PixelLayout = PixelLayout {
    bpp: 4,
    r_off: 0,
    g_off: 1,
    b_off: 2,
    is_cmyk: false,
};
pub const BGRX: PixelLayout = PixelLayout {
    bpp: 4,
    r_off: 2,
    g_off: 1,
    b_off: 0,
    is_cmyk: false,
};
/// Single-byte grayscale layout. The one byte per pixel **is** Y
/// (level-shifted by the caller for the encoder; written verbatim by
/// the decoder). The R/G/B offsets are placeholders — code that runs
/// the YCbCr→RGB color path must branch on `bpp == 1` first; this
/// layout is never passed to the per-arch color kernels.
pub const GRAY: PixelLayout = PixelLayout {
    bpp: 1,
    r_off: 0,
    g_off: 0,
    b_off: 0,
    is_cmyk: false,
};
/// Four-byte CMYK pass-through layout (C, M, Y, K in that order). The
/// encoder treats each of the four channels as an independent
/// component and emits a 4-component baseline JPEG without any
/// CMYK↔RGB transform; the decoder reads such a JPEG back into the
/// same byte order. The R/G/B offsets are placeholders — like
/// [`GRAY`], CMYK never reaches the per-arch color kernels. Callers
/// detect this layout via [`PixelLayout::class`].
pub const CMYK: PixelLayout = PixelLayout {
    bpp: 4,
    r_off: 0,
    g_off: 0,
    b_off: 0,
    is_cmyk: true,
};

/// Build an 8- or 16-wide RGB scratch row with edge replication, starting
/// at pixel column `x0` and source row `sy`. Source row is clamped to
/// `[0, height)` by the caller.
fn build_padded_row<const N: usize>(
    pixels: &[u8],
    width: u32,
    layout: PixelLayout,
    x0: u32,
    sy: usize,
    out: &mut [u8; N],
) where
    [u8; N]: Sized,
{
    let bpp = layout.bpp;
    let pixels_per_row = N / bpp;
    let stride = width as usize * bpp;
    let row_off = sy * stride;
    let max_x = (width - 1) as usize;
    for i in 0..pixels_per_row {
        let sx = (x0 as usize + i).min(max_x);
        let src = row_off + sx * bpp;
        let dst = i * bpp;
        out[dst] = pixels[src];
        out[dst + 1] = pixels[src + 1];
        out[dst + 2] = pixels[src + 2];
        if bpp == 4 {
            out[dst + 3] = pixels[src + 3];
        }
    }
}

/// Extract an 8x8 block of (Y, Cb, Cr) samples, level-shifted to i16.
#[allow(clippy::too_many_arguments)]
pub fn extract_block_ycbcr(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    x0: u32,
    y0: u32,
    y_block: &mut [i16; 64],
    cb_block: &mut [i16; 64],
    cr_block: &mut [i16; 64],
) {
    let max_y = (height - 1) as usize;
    let pixels_per_row_full = width as usize;
    let bpp = layout.bpp;
    let stride = pixels_per_row_full * bpp;

    let mut y_row = [0u8; 8];
    let mut cb_row = [0u8; 8];
    let mut cr_row = [0u8; 8];

    let needs_h_pad = (x0 as usize + 8) > pixels_per_row_full;
    let mut padded = [0u8; 8 * 4];

    for j in 0..8 {
        let sy = (y0 as usize + j).min(max_y);
        let row_off = sy * stride;
        let src_slice = if needs_h_pad {
            build_padded_row::<{ 8 * 4 }>(pixels, width, layout, x0, sy, &mut padded);
            &padded[..8 * bpp]
        } else {
            &pixels[row_off + x0 as usize * bpp..row_off + (x0 as usize + 8) * bpp]
        };
        arch::backend::color::rgb_row_to_ycc(
            src_slice,
            layout,
            8,
            &mut y_row,
            &mut cb_row,
            &mut cr_row,
        );
        for i in 0..8 {
            let idx = j * 8 + i;
            // Y is in [0,255]; level-shift to [-128,127].
            y_block[idx] = (y_row[i] as i16) - 128;
            // Cb/Cr libjpeg outputs are biased by 128; subtract to
            // center on 0 for the DCT.
            cb_block[idx] = (cb_row[i] as i16) - 128;
            cr_block[idx] = (cr_row[i] as i16) - 128;
        }
    }
}

/// Extract a full 8x8 RGB block for 4:4:4 without padding/layout checks.
///
/// Caller guarantees `x0 + 8 <= width` and `y0 + 8 <= height`.
#[allow(clippy::too_many_arguments)]
pub fn extract_block_ycbcr_rgb_full(
    pixels: &[u8],
    width: u32,
    x0: u32,
    y0: u32,
    y_block: &mut [i16; 64],
    cb_block: &mut [i16; 64],
    cr_block: &mut [i16; 64],
) {
    debug_assert!(x0 + 8 <= width);
    let stride = width as usize * RGB.bpp;
    let mut y_row = [0u8; 8];
    let mut cb_row = [0u8; 8];
    let mut cr_row = [0u8; 8];

    for j in 0..8 {
        let row_off = (y0 as usize + j) * stride;
        let start = row_off + x0 as usize * RGB.bpp;
        let src_slice = &pixels[start..start + 8 * RGB.bpp];
        arch::backend::color::rgb_row_to_ycc(
            src_slice,
            RGB,
            8,
            &mut y_row,
            &mut cb_row,
            &mut cr_row,
        );
        for i in 0..8 {
            let idx = j * 8 + i;
            y_block[idx] = (y_row[i] as i16) - 128;
            cb_block[idx] = (cb_row[i] as i16) - 128;
            cr_block[idx] = (cr_row[i] as i16) - 128;
        }
    }
}

/// Extract a 16x16 luma window with 4 luma blocks plus 4:2:0 chroma.
#[allow(clippy::too_many_arguments)]
pub fn extract_mcu_420(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    x0: u32,
    y0: u32,
    y_blocks: &mut [[i16; 64]; 4],
    cb_block: &mut [i16; 64],
    cr_block: &mut [i16; 64],
) {
    let max_y = (height - 1) as usize;
    let pixels_per_row_full = width as usize;
    let bpp = layout.bpp;
    let stride = pixels_per_row_full * bpp;

    let mut y_full = [0u8; 16 * 16];
    let mut cb_full = [0u8; 16 * 16];
    let mut cr_full = [0u8; 16 * 16];

    let needs_h_pad = (x0 as usize + 16) > pixels_per_row_full;
    let mut padded = [0u8; 16 * 4];

    for j in 0..16 {
        let sy = (y0 as usize + j).min(max_y);
        let row_off = sy * stride;
        let src_slice = if needs_h_pad {
            build_padded_row::<{ 16 * 4 }>(pixels, width, layout, x0, sy, &mut padded);
            &padded[..16 * bpp]
        } else {
            &pixels[row_off + x0 as usize * bpp..row_off + (x0 as usize + 16) * bpp]
        };
        let off = j * 16;
        arch::backend::color::rgb_row_to_ycc(
            src_slice,
            layout,
            16,
            &mut y_full[off..off + 16],
            &mut cb_full[off..off + 16],
            &mut cr_full[off..off + 16],
        );
    }

    // Distribute luma into the four 8x8 quadrants with level shift.
    for jq in 0..2 {
        for iq in 0..2 {
            let dst = &mut y_blocks[jq * 2 + iq];
            for j in 0..8 {
                for i in 0..8 {
                    dst[j * 8 + i] = (y_full[(jq * 8 + j) * 16 + (iq * 8 + i)] as i16) - 128;
                }
            }
        }
    }

    arch::backend::color::h2v2_downsample(&cb_full, cb_block);
    arch::backend::color::h2v2_downsample(&cr_full, cr_block);
}

/// Extract a full 16x16 RGB MCU for 4:2:0 without padding/layout checks.
///
/// Caller guarantees `x0 + 16 <= width` and that `y0 + 16 <= height`.
/// Edge MCUs should use [`extract_mcu_420`] so right/bottom replication
/// stays centralized in the generic path.
#[allow(clippy::too_many_arguments)]
pub fn extract_mcu_420_rgb_full(
    pixels: &[u8],
    width: u32,
    x0: u32,
    y0: u32,
    y_blocks: &mut [[i16; 64]; 4],
    cb_block: &mut [i16; 64],
    cr_block: &mut [i16; 64],
) {
    debug_assert!(x0 + 16 <= width);
    let stride = width as usize * RGB.bpp;
    let mut y_full = [0u8; 16 * 16];
    let mut cb_full = [0u8; 16 * 16];
    let mut cr_full = [0u8; 16 * 16];

    for j in 0..16 {
        let row_off = (y0 as usize + j) * stride;
        let start = row_off + x0 as usize * RGB.bpp;
        let src_slice = &pixels[start..start + 16 * RGB.bpp];
        let off = j * 16;
        arch::backend::color::rgb_row_to_ycc(
            src_slice,
            RGB,
            16,
            &mut y_full[off..off + 16],
            &mut cb_full[off..off + 16],
            &mut cr_full[off..off + 16],
        );
    }

    for jq in 0..2 {
        for iq in 0..2 {
            let dst = &mut y_blocks[jq * 2 + iq];
            for j in 0..8 {
                for i in 0..8 {
                    dst[j * 8 + i] = (y_full[(jq * 8 + j) * 16 + (iq * 8 + i)] as i16) - 128;
                }
            }
        }
    }

    arch::backend::color::h2v2_downsample(&cb_full, cb_block);
    arch::backend::color::h2v2_downsample(&cr_full, cr_block);
}

/// Extract a 16x8 luma window with 2 luma blocks plus 4:2:2 chroma.
///
/// MCU layout: two horizontally-adjacent 8x8 luma blocks (left, right)
/// and one 8x8 chroma block per Cb/Cr produced by 2:1 horizontal
/// downsample of the 16-wide chroma row pair.
#[allow(clippy::too_many_arguments)]
pub fn extract_mcu_422(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    x0: u32,
    y0: u32,
    y_blocks: &mut [[i16; 64]; 2],
    cb_block: &mut [i16; 64],
    cr_block: &mut [i16; 64],
) {
    let max_y = (height - 1) as usize;
    let pixels_per_row_full = width as usize;
    let bpp = layout.bpp;
    let stride = pixels_per_row_full * bpp;

    let mut y_full = [0u8; 16 * 8];
    let mut cb_full = [0u8; 16 * 8];
    let mut cr_full = [0u8; 16 * 8];

    let needs_h_pad = (x0 as usize + 16) > pixels_per_row_full;
    let mut padded = [0u8; 16 * 4];

    for j in 0..8 {
        let sy = (y0 as usize + j).min(max_y);
        let row_off = sy * stride;
        let src_slice = if needs_h_pad {
            build_padded_row::<{ 16 * 4 }>(pixels, width, layout, x0, sy, &mut padded);
            &padded[..16 * bpp]
        } else {
            &pixels[row_off + x0 as usize * bpp..row_off + (x0 as usize + 16) * bpp]
        };
        let off = j * 16;
        arch::backend::color::rgb_row_to_ycc(
            src_slice,
            layout,
            16,
            &mut y_full[off..off + 16],
            &mut cb_full[off..off + 16],
            &mut cr_full[off..off + 16],
        );
    }

    // Distribute luma into the two 8x8 halves with level shift.
    for iq in 0..2 {
        let dst = &mut y_blocks[iq];
        for j in 0..8 {
            for i in 0..8 {
                dst[j * 8 + i] = (y_full[j * 16 + (iq * 8 + i)] as i16) - 128;
            }
        }
    }

    arch::backend::color::h2v1_downsample(&cb_full, cb_block);
    arch::backend::color::h2v1_downsample(&cr_full, cr_block);
}

/// Extract a full 16x8 RGB MCU for 4:2:2 without padding/layout checks.
///
/// Caller guarantees `x0 + 16 <= width` and `y0 + 8 <= height`.
#[allow(clippy::too_many_arguments)]
pub fn extract_mcu_422_rgb_full(
    pixels: &[u8],
    width: u32,
    x0: u32,
    y0: u32,
    y_blocks: &mut [[i16; 64]; 2],
    cb_block: &mut [i16; 64],
    cr_block: &mut [i16; 64],
) {
    debug_assert!(x0 + 16 <= width);
    let stride = width as usize * RGB.bpp;
    let mut y_full = [0u8; 16 * 8];
    let mut cb_full = [0u8; 16 * 8];
    let mut cr_full = [0u8; 16 * 8];

    for j in 0..8 {
        let row_off = (y0 as usize + j) * stride;
        let start = row_off + x0 as usize * RGB.bpp;
        let src_slice = &pixels[start..start + 16 * RGB.bpp];
        let off = j * 16;
        arch::backend::color::rgb_row_to_ycc(
            src_slice,
            RGB,
            16,
            &mut y_full[off..off + 16],
            &mut cb_full[off..off + 16],
            &mut cr_full[off..off + 16],
        );
    }

    for (iq, dst) in y_blocks.iter_mut().enumerate() {
        for j in 0..8 {
            for i in 0..8 {
                dst[j * 8 + i] = (y_full[j * 16 + (iq * 8 + i)] as i16) - 128;
            }
        }
    }

    arch::backend::color::h2v1_downsample(&cb_full, cb_block);
    arch::backend::color::h2v1_downsample(&cr_full, cr_block);
}

/// Extract an 8x8 block of one CMYK channel (selected by `channel`,
/// `0..=3` for C/M/Y/K) from a 4-byte/pixel CMYK buffer,
/// level-shifted to centered i16. Edge-replicates on the right and
/// bottom borders, matching `extract_block_gray`'s shape.
///
/// CMYK encode treats each of the four channels as an independent
/// component (sampling factor 1:1:1:1, shared luma quant / Huffman),
/// so the per-MCU pass calls this fn four times — one block per
/// channel.
pub fn extract_block_cmyk(
    pixels: &[u8],
    width: u32,
    height: u32,
    x0: u32,
    y0: u32,
    channel: usize,
    block: &mut [i16; 64],
) {
    debug_assert!(channel < 4);
    let max_x = (width - 1) as usize;
    let max_y = (height - 1) as usize;
    let stride = width as usize * 4;
    for j in 0..8 {
        let sy = (y0 as usize + j).min(max_y);
        let row_off = sy * stride;
        for i in 0..8 {
            let sx = (x0 as usize + i).min(max_x);
            // Level-shift unsigned [0,255] to signed [-128,127] for the
            // forward DCT (T.81 A.3.1).
            block[j * 8 + i] = (pixels[row_off + sx * 4 + channel] as i16) - 128;
        }
    }
}

/// Extract an 8x8 block of Y samples from a 1-byte/pixel grayscale
/// buffer, level-shifted to centered i16. Edge-replicates on the right
/// and bottom borders (the encoder always feeds 8-pixel-aligned MCUs
/// even when the image is not a multiple of 8).
///
/// Unlike [`extract_block_ycbcr`], this fn skips the RGB→YCbCr SIMD
/// kernel entirely — the input byte already *is* Y.
pub fn extract_block_gray(
    pixels: &[u8],
    width: u32,
    height: u32,
    x0: u32,
    y0: u32,
    y_block: &mut [i16; 64],
) {
    let max_x = (width - 1) as usize;
    let max_y = (height - 1) as usize;
    let stride = width as usize;
    for j in 0..8 {
        let sy = (y0 as usize + j).min(max_y);
        let row_off = sy * stride;
        for i in 0..8 {
            let sx = (x0 as usize + i).min(max_x);
            // Level-shift unsigned [0,255] to signed [-128,127] for the
            // forward DCT (T.81 A.3.1).
            y_block[j * 8 + i] = (pixels[row_off + sx] as i16) - 128;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned_rgb(width: usize, height: usize) -> Vec<u8> {
        let mut pixels = vec![0u8; width * height * RGB.bpp];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * RGB.bpp;
                pixels[i] = ((x * 17 + y * 3) & 0xFF) as u8;
                pixels[i + 1] = ((x * 5 + y * 11 + 7) & 0xFF) as u8;
                pixels[i + 2] = ((x * 13 + y * 19 + 23) & 0xFF) as u8;
            }
        }
        pixels
    }

    #[test]
    fn extract_block_ycbcr_rgb_full_matches_generic_rgb() {
        let width = 24;
        let height = 24;
        let pixels = patterned_rgb(width, height);

        let mut y_generic = [0i16; 64];
        let mut cb_generic = [0i16; 64];
        let mut cr_generic = [0i16; 64];
        extract_block_ycbcr(
            &pixels,
            width as u32,
            height as u32,
            RGB,
            8,
            8,
            &mut y_generic,
            &mut cb_generic,
            &mut cr_generic,
        );

        let mut y_fast = [0i16; 64];
        let mut cb_fast = [0i16; 64];
        let mut cr_fast = [0i16; 64];
        extract_block_ycbcr_rgb_full(
            &pixels,
            width as u32,
            8,
            8,
            &mut y_fast,
            &mut cb_fast,
            &mut cr_fast,
        );

        assert_eq!(y_fast, y_generic);
        assert_eq!(cb_fast, cb_generic);
        assert_eq!(cr_fast, cr_generic);
    }

    #[test]
    fn extract_mcu_420_rgb_full_matches_generic_rgb() {
        let width = 32;
        let height = 32;
        let pixels = patterned_rgb(width, height);

        let mut y_generic = [[0i16; 64]; 4];
        let mut cb_generic = [0i16; 64];
        let mut cr_generic = [0i16; 64];
        extract_mcu_420(
            &pixels,
            width as u32,
            height as u32,
            RGB,
            16,
            16,
            &mut y_generic,
            &mut cb_generic,
            &mut cr_generic,
        );

        let mut y_fast = [[0i16; 64]; 4];
        let mut cb_fast = [0i16; 64];
        let mut cr_fast = [0i16; 64];
        extract_mcu_420_rgb_full(
            &pixels,
            width as u32,
            16,
            16,
            &mut y_fast,
            &mut cb_fast,
            &mut cr_fast,
        );

        assert_eq!(y_fast, y_generic);
        assert_eq!(cb_fast, cb_generic);
        assert_eq!(cr_fast, cr_generic);
    }

    #[test]
    fn extract_mcu_422_rgb_full_matches_generic_rgb() {
        let width = 32;
        let height = 16;
        let pixels = patterned_rgb(width, height);

        let mut y_generic = [[0i16; 64]; 2];
        let mut cb_generic = [0i16; 64];
        let mut cr_generic = [0i16; 64];
        extract_mcu_422(
            &pixels,
            width as u32,
            height as u32,
            RGB,
            16,
            8,
            &mut y_generic,
            &mut cb_generic,
            &mut cr_generic,
        );

        let mut y_fast = [[0i16; 64]; 2];
        let mut cb_fast = [0i16; 64];
        let mut cr_fast = [0i16; 64];
        extract_mcu_422_rgb_full(
            &pixels,
            width as u32,
            16,
            8,
            &mut y_fast,
            &mut cb_fast,
            &mut cr_fast,
        );

        assert_eq!(y_fast, y_generic);
        assert_eq!(cb_fast, cb_generic);
        assert_eq!(cr_fast, cr_generic);
    }
}
