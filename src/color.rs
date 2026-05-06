//! YCbCr block extraction with edge replication.
//!
//! The per-row RGB→YCbCr and 16x16→8x8 chroma downsample kernels live
//! in `crate::arch::backend::color`. This file orchestrates them at the
//! 8x8 block / 4:2:0 MCU level — padding, level shift, layout
//! distribution.
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

/// Pixel-stride descriptor: bytes per pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelLayout {
    pub bpp: usize,
}

pub const RGB: PixelLayout = PixelLayout { bpp: 3 };
pub const RGBA: PixelLayout = PixelLayout { bpp: 4 };

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
