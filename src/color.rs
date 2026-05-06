//! RGB / RGBA → YCbCr block extraction with edge replication.
//!
//! Phase 2 outputs are `i16` already-level-shifted samples (Y in
//! `[-128, 127]`, Cb/Cr also centered on 0) ready for the integer LL&M
//! DCT. Cb/Cr in libjpeg-turbo's `jccolext-neon.c` are produced
//! centered on 128 (i.e. unsigned `[0, 255]`); we subtract 128 inline
//! so the DCT input contract is uniform across components.
//!
//! The hot path on aarch64 calls into the NEON kernel one row at a
//! time. Edge replication for non-multiple-of-MCU images happens in a
//! 16- or 8-wide scratch buffer (same trick libjpeg uses).
//!
//! Algorithmic constants follow libjpeg-turbo's `jccolor-neon.c`:
//!   Y  =  0.29900 R + 0.58700 G + 0.11400 B
//!   Cb = -0.16874 R - 0.33126 G + 0.50000 B + 128
//!   Cr =  0.50000 R - 0.41869 G - 0.08131 B + 128
//! all encoded as 16-bit fixed-point and descaled by 16 after summing.

// The scalar `rgb_row_to_ycc_scalar` and `h2v2_scalar` are reachable on
// aarch64 only from the unit tests. Same for some constants that the
// scalar path uses individually but the NEON path bundles into a
// constant table. Suppress unused-warnings rather than littering each
// item with `#[allow(...)]`.
#![allow(dead_code)]

const FIX: i32 = 16;
const FIX_HALF: i32 = 1 << (FIX - 1);

const Y_R: u32 = 19595;
const Y_G: u32 = 38470;
const Y_B: u32 = 7471;

const CB_R: u32 = 11059;
const CB_G: u32 = 21709;
const CB_B: u32 = 32768;

const CR_R: u32 = 32768;
const CR_G: u32 = 27439;
const CR_B: u32 = 5329;

// "0.5 + 128 in fixed-point Cb/Cr space" — see libjpeg jccolext-neon.c:
//   scaled_128_5 = (128 << 16) + 32767
// The +32767 (not 32768) is the rounding bias rolled into the +128
// constant; vshrn_n_u32(x, 16) is a non-rounding shift, so we have to
// pre-bias once.
const CHROMA_BIAS: u32 = (128 << 16) + 32767;

/// Pixel-stride descriptor: bytes per pixel.
#[derive(Clone, Copy)]
pub struct PixelLayout {
    pub bpp: usize,
}

pub const RGB: PixelLayout = PixelLayout { bpp: 3 };
pub const RGBA: PixelLayout = PixelLayout { bpp: 4 };

/// Convert one row of length `n` (n must be a multiple of 8 in the
/// scalar path; the caller pads the trailing partial-row in a scratch
/// buffer). Outputs Y, Cb, Cr as `u8` (centered on 128 for chroma) —
/// this matches the libjpeg-turbo NEON kernel's contract so the SIMD
/// version is a drop-in replacement. Level shift to i16 happens later.
fn rgb_row_to_ycc_scalar(
    pixels: &[u8],
    layout: PixelLayout,
    n: usize,
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
) {
    debug_assert!(y.len() >= n && cb.len() >= n && cr.len() >= n);
    debug_assert!(pixels.len() >= n * layout.bpp);
    for i in 0..n {
        let p = i * layout.bpp;
        let r = pixels[p] as u32;
        let g = pixels[p + 1] as u32;
        let b = pixels[p + 2] as u32;
        let y_i = (Y_R * r + Y_G * g + Y_B * b + FIX_HALF as u32) >> FIX;
        let cb_i = (CHROMA_BIAS - CB_R * r - CB_G * g + CB_B * b) >> FIX;
        let cr_i = (CHROMA_BIAS + CR_R * r - CR_G * g - CR_B * b) >> FIX;
        y[i] = y_i as u8;
        cb[i] = cb_i as u8;
        cr[i] = cr_i as u8;
    }
}

#[cfg(target_arch = "aarch64")]
mod color_neon {
    use core::arch::aarch64::*;

    /// Constants laid out for `vmull_laneq_u16` indexing 0..=7:
    ///   0: 0.299 R weight (Y)
    ///   1: 0.587 G weight (Y)
    ///   2: 0.114 B weight (Y)
    ///   3: 0.16874 R weight (Cb, subtracted)
    ///   4: 0.33126 G weight (Cb, subtracted)
    ///   5: 0.50000 R/B weight (Cb adds B, Cr adds R)
    ///   6: 0.41869 G weight (Cr, subtracted)
    ///   7: 0.08131 B weight (Cr, subtracted)
    const CONSTS: [u16; 8] = [19595, 38470, 7471, 11059, 21709, 32768, 27439, 5329];

    /// Convert exactly 16 RGB(A) pixels to Y/Cb/Cr. Caller guarantees
    /// readable input and writable outputs of length 16.
    #[target_feature(enable = "neon")]
    unsafe fn rgb16_to_ycc(
        consts: uint16x8_t,
        chroma_bias: uint32x4_t,
        r: uint16x8_t,
        g: uint16x8_t,
        b: uint16x8_t,
    ) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
        // Y = sum * 0.299/0.587/0.114, rounding shr 16.
        let mut y_l = vmull_laneq_u16::<0>(vget_low_u16(r), consts);
        y_l = vmlal_laneq_u16::<1>(y_l, vget_low_u16(g), consts);
        y_l = vmlal_laneq_u16::<2>(y_l, vget_low_u16(b), consts);
        let mut y_h = vmull_laneq_u16::<0>(vget_high_u16(r), consts);
        y_h = vmlal_laneq_u16::<1>(y_h, vget_high_u16(g), consts);
        y_h = vmlal_laneq_u16::<2>(y_h, vget_high_u16(b), consts);

        // Cb = bias - 0.16874 R - 0.33126 G + 0.5 B (truncating shr 16).
        let mut cb_l = chroma_bias;
        cb_l = vmlsl_laneq_u16::<3>(cb_l, vget_low_u16(r), consts);
        cb_l = vmlsl_laneq_u16::<4>(cb_l, vget_low_u16(g), consts);
        cb_l = vmlal_laneq_u16::<5>(cb_l, vget_low_u16(b), consts);
        let mut cb_h = chroma_bias;
        cb_h = vmlsl_laneq_u16::<3>(cb_h, vget_high_u16(r), consts);
        cb_h = vmlsl_laneq_u16::<4>(cb_h, vget_high_u16(g), consts);
        cb_h = vmlal_laneq_u16::<5>(cb_h, vget_high_u16(b), consts);

        // Cr = bias + 0.5 R - 0.41869 G - 0.08131 B.
        let mut cr_l = chroma_bias;
        cr_l = vmlal_laneq_u16::<5>(cr_l, vget_low_u16(r), consts);
        cr_l = vmlsl_laneq_u16::<6>(cr_l, vget_low_u16(g), consts);
        cr_l = vmlsl_laneq_u16::<7>(cr_l, vget_low_u16(b), consts);
        let mut cr_h = chroma_bias;
        cr_h = vmlal_laneq_u16::<5>(cr_h, vget_high_u16(r), consts);
        cr_h = vmlsl_laneq_u16::<6>(cr_h, vget_high_u16(g), consts);
        cr_h = vmlsl_laneq_u16::<7>(cr_h, vget_high_u16(b), consts);

        // Y uses rounding narrow; Cb/Cr use truncating narrow because
        // the +32767 is already in the bias.
        let y_u16 = vcombine_u16(vrshrn_n_u32::<16>(y_l), vrshrn_n_u32::<16>(y_h));
        let cb_u16 = vcombine_u16(vshrn_n_u32::<16>(cb_l), vshrn_n_u32::<16>(cb_h));
        let cr_u16 = vcombine_u16(vshrn_n_u32::<16>(cr_l), vshrn_n_u32::<16>(cr_h));
        (y_u16, cb_u16, cr_u16)
    }

    /// Convert exactly 16 pixels at `inptr` (any of bpp 3 or 4) into 16
    /// Y/Cb/Cr bytes at the output pointers.
    #[target_feature(enable = "neon")]
    pub unsafe fn rgb_row_16(
        inptr: *const u8,
        bpp: usize,
        outy: *mut u8,
        outcb: *mut u8,
        outcr: *mut u8,
    ) { unsafe {
        let consts = vld1q_u16(CONSTS.as_ptr());
        let bias = vdupq_n_u32(super::CHROMA_BIAS);
        let (r, g, b) = if bpp == 4 {
            let p = vld4q_u8(inptr);
            (p.0, p.1, p.2)
        } else {
            let p = vld3q_u8(inptr);
            (p.0, p.1, p.2)
        };
        let r_l = vmovl_u8(vget_low_u8(r));
        let g_l = vmovl_u8(vget_low_u8(g));
        let b_l = vmovl_u8(vget_low_u8(b));
        let r_h = vmovl_u8(vget_high_u8(r));
        let g_h = vmovl_u8(vget_high_u8(g));
        let b_h = vmovl_u8(vget_high_u8(b));
        let (y_l, cb_l, cr_l) = rgb16_to_ycc(consts, bias, r_l, g_l, b_l);
        let (y_h, cb_h, cr_h) = rgb16_to_ycc(consts, bias, r_h, g_h, b_h);
        vst1q_u8(outy, vcombine_u8(vmovn_u16(y_l), vmovn_u16(y_h)));
        vst1q_u8(outcb, vcombine_u8(vmovn_u16(cb_l), vmovn_u16(cb_h)));
        vst1q_u8(outcr, vcombine_u8(vmovn_u16(cr_l), vmovn_u16(cr_h)));
    }}

    /// 8-pixel variant for 8-wide scratch rows.
    #[target_feature(enable = "neon")]
    pub unsafe fn rgb_row_8(
        inptr: *const u8,
        bpp: usize,
        outy: *mut u8,
        outcb: *mut u8,
        outcr: *mut u8,
    ) { unsafe {
        let consts = vld1q_u16(CONSTS.as_ptr());
        let bias = vdupq_n_u32(super::CHROMA_BIAS);
        let (r, g, b) = if bpp == 4 {
            let p = vld4_u8(inptr);
            (p.0, p.1, p.2)
        } else {
            let p = vld3_u8(inptr);
            (p.0, p.1, p.2)
        };
        let r = vmovl_u8(r);
        let g = vmovl_u8(g);
        let b = vmovl_u8(b);
        let (y, cb, cr) = rgb16_to_ycc(consts, bias, r, g, b);
        vst1_u8(outy, vmovn_u16(y));
        vst1_u8(outcb, vmovn_u16(cb));
        vst1_u8(outcr, vmovn_u16(cr));
    }}
}

/// Convert one row of `n` pixels to YCbCr (chroma centered on 128).
fn rgb_row_to_ycc(
    pixels: &[u8],
    layout: PixelLayout,
    n: usize,
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
) {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    unsafe {
        // NEON is mandatory on aarch64. The block sizes we're called
        // with (8, 16) hit the fast path exactly.
        debug_assert!(n == 8 || n == 16);
        if n == 16 {
            color_neon::rgb_row_16(
                pixels.as_ptr(),
                layout.bpp,
                y.as_mut_ptr(),
                cb.as_mut_ptr(),
                cr.as_mut_ptr(),
            );
        } else {
            color_neon::rgb_row_8(
                pixels.as_ptr(),
                layout.bpp,
                y.as_mut_ptr(),
                cb.as_mut_ptr(),
                cr.as_mut_ptr(),
            );
        }
    }
    #[cfg(not(all(target_arch = "aarch64", not(feature = "force-scalar"))))]
    rgb_row_to_ycc_scalar(pixels, layout, n, y, cb, cr);
}

/// Build an 8-wide RGB scratch row with edge replication, starting at
/// pixel column `x0` and source row `sy`. Source row is clamped to
/// [0, height) by the caller.
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
    // N must be 8 * bpp or 16 * bpp.
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

    // Direct slice into source if we don't need horizontal padding.
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
        rgb_row_to_ycc(src_slice, layout, 8, &mut y_row, &mut cb_row, &mut cr_row);
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
        rgb_row_to_ycc(
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
                    dst[j * 8 + i] =
                        (y_full[(jq * 8 + j) * 16 + (iq * 8 + i)] as i16) - 128;
                }
            }
        }
    }

    // 2x2 box average chroma into 8x8 blocks. We use libjpeg-turbo's
    // bias pattern for 2:1+2:1 ("standard" downsample): bias of 1 on
    // even columns, 2 on odd columns within each 4-sample group, then
    // shift right by 2. (This is what `jsimd_h2v2_downsample_neon` does
    // with `vpadalq_u8(bias=0x00020001-broadcast)`. We do it as a
    // straight scalar pass; a separate NEON kernel below covers the
    // common case.) Result is unsigned [0,255]; level-shift after.
    chroma_h2v2_downsample(&cb_full, cb_block);
    chroma_h2v2_downsample(&cr_full, cr_block);
}

/// Downsample a 16x16 plane to 8x8 using libjpeg-turbo's biased 2x2
/// average. Output is level-shifted to i16 centered on 0.
fn chroma_h2v2_downsample(src: &[u8; 256], dst: &mut [i16; 64]) {
    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    unsafe { downsample_neon::h2v2(src, dst) };
    #[cfg(not(all(target_arch = "aarch64", not(feature = "force-scalar"))))]
    h2v2_scalar(src, dst);
}

#[allow(dead_code)]
fn h2v2_scalar(src: &[u8; 256], dst: &mut [i16; 64]) {
    // libjpeg-turbo's bias table is {1, 2, 1, 2, ...} broadcast over the
    // 8 output u16 lanes per row pair. The bias is added once via
    // `vpadalq_u8`, so the rounding offset alternates per *output* column:
    // even output column → +1, odd output column → +2. This means a
    // half-pixel-shift in chroma siting that libjpeg considers acceptable
    // and that we must reproduce for bit-exact equivalence.
    for j in 0..8 {
        for i in 0..8 {
            let r0 = j * 2;
            let r1 = r0 + 1;
            let c0 = i * 2;
            let c1 = c0 + 1;
            let bias = if i % 2 == 0 { 1u32 } else { 2 };
            let r = src[r0 * 16 + c0] as u32
                + src[r0 * 16 + c1] as u32
                + src[r1 * 16 + c0] as u32
                + src[r1 * 16 + c1] as u32;
            let avg = (r + bias) >> 2;
            dst[j * 8 + i] = (avg as i16) - 128;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod downsample_neon {
    use core::arch::aarch64::*;

    /// 16x16 → 8x8 box average with libjpeg-turbo's pairwise bias
    /// pattern. The +3 rounding bias is split as +1 on row 0, +2 on
    /// row 1, distributed across alternating 16-bit lanes, exactly as
    /// `jsimd_h2v2_downsample_neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn h2v2(src: &[u8; 256], dst: &mut [i16; 64]) { unsafe {
        // Row 0 bias { 1, 0, 1, 0, ... } and row 1 bias { 0, 2, 0, 2, ... }
        // combined into one row pair = { 1, 2, 1, 2, ... } over 16-bit lanes.
        let bias = vreinterpretq_u16_u32(vdupq_n_u32(0x0002_0001));
        let level_shift = vdupq_n_s16(128);
        for j in 0..8 {
            let row0_off = j * 2 * 16;
            let row1_off = row0_off + 16;
            let r0 = vld1q_u8(src.as_ptr().add(row0_off));
            let r1 = vld1q_u8(src.as_ptr().add(row1_off));
            // pairwise add (vpadalq_u8 = bias + sum of pairs).
            let sums = vpadalq_u8(bias, r0);
            let sums = vpadalq_u8(sums, r1);
            // Divide by 4; narrow to 8-bit; widen to i16 and level-shift.
            let avg_u8 = vshrn_n_u16::<2>(sums);
            let avg_u16 = vmovl_u8(avg_u8);
            let signed = vsubq_s16(vreinterpretq_s16_u16(avg_u16), level_shift);
            vst1q_s16(dst.as_mut_ptr().add(j * 8), signed);
        }
    }}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_pixel_yields_minus_128_y() {
        let pixels = [0u8; 8 * 4];
        let mut y_row = [0u8; 8];
        let mut cb_row = [0u8; 8];
        let mut cr_row = [0u8; 8];
        rgb_row_to_ycc(&pixels, RGBA, 8, &mut y_row, &mut cb_row, &mut cr_row);
        for i in 0..8 {
            assert_eq!(y_row[i], 0);
            // Cb/Cr for pure black (R=G=B=0) is exactly 128.
            assert_eq!(cb_row[i], 128);
            assert_eq!(cr_row[i], 128);
        }
    }

    #[test]
    fn white_pixel_yields_255_y() {
        let pixels = [255u8; 8 * 4];
        let mut y_row = [0u8; 8];
        let mut cb_row = [0u8; 8];
        let mut cr_row = [0u8; 8];
        rgb_row_to_ycc(&pixels, RGBA, 8, &mut y_row, &mut cb_row, &mut cr_row);
        for i in 0..8 {
            assert_eq!(y_row[i], 255);
            assert_eq!(cb_row[i], 128);
            assert_eq!(cr_row[i], 128);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_color() {
        // Deterministic-ish input: gradient + alternating pattern.
        let mut pixels = [0u8; 16 * 4];
        for i in 0..16 {
            pixels[i * 4] = (i * 17) as u8;
            pixels[i * 4 + 1] = ((i * 23 + 7) % 256) as u8;
            pixels[i * 4 + 2] = ((i * 31 + 13) % 256) as u8;
            pixels[i * 4 + 3] = 255;
        }
        let mut y_s = [0u8; 16];
        let mut cb_s = [0u8; 16];
        let mut cr_s = [0u8; 16];
        rgb_row_to_ycc_scalar(&pixels, RGBA, 16, &mut y_s, &mut cb_s, &mut cr_s);

        let mut y_n = [0u8; 16];
        let mut cb_n = [0u8; 16];
        let mut cr_n = [0u8; 16];
        unsafe {
            color_neon::rgb_row_16(
                pixels.as_ptr(),
                4,
                y_n.as_mut_ptr(),
                cb_n.as_mut_ptr(),
                cr_n.as_mut_ptr(),
            );
        }
        assert_eq!(y_s, y_n);
        assert_eq!(cb_s, cb_n);
        assert_eq!(cr_s, cr_n);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_downsample() {
        let mut src = [0u8; 256];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i * 53 + 17) % 256) as u8;
        }
        let mut a = [0i16; 64];
        let mut b = [0i16; 64];
        h2v2_scalar(&src, &mut a);
        unsafe { downsample_neon::h2v2(&src, &mut b) };
        assert_eq!(a, b);
    }
}
