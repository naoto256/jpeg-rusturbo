//! Architecture-independent reference implementations of the hot
//! kernels: per-row RGB→YCbCr, 4:2:0 / 4:2:2 chroma downsample,
//! integer LL&M FDCT, and reciprocal-multiply quantization (with the
//! small Huffman AC-zero-scan helper rounding things out).
//!
//! Every NEON / x86_64 backend produces bit-exact output against this
//! file. When the `force-scalar` feature is on (or the target arch has
//! no SIMD backend), this module is selected as `arch::backend`.

#![allow(dead_code)]

// ===========================================================================
// color: RGB(A) → YCbCr (per-row), chroma downsample (4:2:0 and 4:2:2)
// ===========================================================================
pub mod color {
    use crate::color::{
        CB_B, CB_G, CB_R, CHROMA_BIAS, CR_B, CR_G, CR_R, FIX, FIX_HALF, PixelLayout, Y_B, Y_G, Y_R,
    };

    /// Convert one row of `n` pixels (multiple of 8) to YCbCr (chroma
    /// centered on 128). Output is unsigned `u8`; level shift to i16
    /// happens at the block-extraction layer.
    pub fn rgb_row_to_ycc(
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

    /// Downsample a 16x16 plane to 8x8 using libjpeg-turbo's biased 2x2
    /// average. Output is level-shifted to i16 centered on 0.
    ///
    /// libjpeg's bias pattern is `{1, 2, 1, 2, ...}` over the 8 output
    /// u16 lanes per row pair, added once and shifted right by 2 — i.e.
    /// rounding +1 on even output columns, +2 on odd. We must reproduce
    /// this for bit-exact equivalence with the NEON path.
    pub fn h2v2_downsample(src: &[u8; 256], dst: &mut [i16; 64]) {
        for j in 0..8 {
            for i in 0..8 {
                let r0 = j * 2;
                let r1 = r0 + 1;
                let c0 = i * 2;
                let c1 = c0 + 1;
                let bias = if i % 2 == 0 { 1u32 } else { 2 };
                let sum = src[r0 * 16 + c0] as u32
                    + src[r0 * 16 + c1] as u32
                    + src[r1 * 16 + c0] as u32
                    + src[r1 * 16 + c1] as u32;
                let avg = (sum + bias) >> 2;
                dst[j * 8 + i] = (avg as i16) - 128;
            }
        }
    }

    /// Downsample a 16x8 plane to 8x8 by 2:1 horizontal averaging, with
    /// libjpeg-turbo's biased rounding. Output is level-shifted to i16
    /// centered on 0.
    ///
    /// Bias alternates `{0, 1, 0, 1, ...}` across output columns, added
    /// once and shifted right by 1. See libjpeg-turbo `jcsample.c`,
    /// `h2v1_downsample`. The SIMD ports must reproduce this for
    /// bit-exact equivalence with this reference.
    pub fn h2v1_downsample(src: &[u8; 128], dst: &mut [i16; 64]) {
        for j in 0..8 {
            for i in 0..8 {
                let c0 = i * 2;
                let c1 = c0 + 1;
                let bias = if i % 2 == 0 { 0u32 } else { 1 };
                let sum = src[j * 16 + c0] as u32 + src[j * 16 + c1] as u32;
                let avg = (sum + bias) >> 1;
                dst[j * 8 + i] = (avg as i16) - 128;
            }
        }
    }
}

// ===========================================================================
// dct: integer LL&M ("islow") FDCT, 12 multiplies per pass, 13-bit consts
// ===========================================================================
pub mod dct {
    const CONST_BITS: i32 = 13;
    const PASS1_BITS: i32 = 2;

    const FIX_0_298_631_336: i32 = 2446;
    const FIX_0_390_180_644: i32 = 3196;
    const FIX_0_541_196_100: i32 = 4433;
    const FIX_0_765_366_865: i32 = 6270;
    const FIX_0_899_976_223: i32 = 7373;
    const FIX_1_175_875_602: i32 = 9633;
    const FIX_1_501_321_110: i32 = 12299;
    const FIX_1_847_759_065: i32 = 15137;
    const FIX_1_961_570_560: i32 = 16069;
    const FIX_2_053_119_869: i32 = 16819;
    const FIX_2_562_915_447: i32 = 20995;
    const FIX_3_072_711_026: i32 = 25172;

    /// Round-and-shift: `(x + 2^(n-1)) >> n`, matching libjpeg's `DESCALE`.
    #[inline(always)]
    fn descale(x: i32, n: i32) -> i32 {
        (x + (1 << (n - 1))) >> n
    }

    /// Forward 8x8 DCT, in place. Input level-shifted samples; output is
    /// 16-bit DCT coefficients scaled by 8 (the libjpeg LL&M convention,
    /// absorbed by quantization downstream).
    pub fn fdct_islow(data: &mut [i16; 64]) {
        // Pass 1: rows. Outputs are scaled up by 2^PASS1_BITS for extra
        // intermediate precision.
        for row in 0..8 {
            let off = row * 8;
            let d0 = data[off] as i32;
            let d1 = data[off + 1] as i32;
            let d2 = data[off + 2] as i32;
            let d3 = data[off + 3] as i32;
            let d4 = data[off + 4] as i32;
            let d5 = data[off + 5] as i32;
            let d6 = data[off + 6] as i32;
            let d7 = data[off + 7] as i32;

            let tmp0 = d0 + d7;
            let tmp7 = d0 - d7;
            let tmp1 = d1 + d6;
            let tmp6 = d1 - d6;
            let tmp2 = d2 + d5;
            let tmp5 = d2 - d5;
            let tmp3 = d3 + d4;
            let tmp4 = d3 - d4;

            let tmp10 = tmp0 + tmp3;
            let tmp13 = tmp0 - tmp3;
            let tmp11 = tmp1 + tmp2;
            let tmp12 = tmp1 - tmp2;

            data[off] = ((tmp10 + tmp11) << PASS1_BITS) as i16;
            data[off + 4] = ((tmp10 - tmp11) << PASS1_BITS) as i16;

            let z1 = (tmp12 + tmp13) * FIX_0_541_196_100;
            data[off + 2] = descale(z1 + tmp13 * FIX_0_765_366_865, CONST_BITS - PASS1_BITS) as i16;
            data[off + 6] =
                descale(z1 + tmp12 * (-FIX_1_847_759_065), CONST_BITS - PASS1_BITS) as i16;

            let z1 = tmp4 + tmp7;
            let z2 = tmp5 + tmp6;
            let z3 = tmp4 + tmp6;
            let z4 = tmp5 + tmp7;
            let z5 = (z3 + z4) * FIX_1_175_875_602;

            let t4 = tmp4 * FIX_0_298_631_336;
            let t5 = tmp5 * FIX_2_053_119_869;
            let t6 = tmp6 * FIX_3_072_711_026;
            let t7 = tmp7 * FIX_1_501_321_110;
            let z1 = z1 * (-FIX_0_899_976_223);
            let z2 = z2 * (-FIX_2_562_915_447);
            let z3 = z3 * (-FIX_1_961_570_560);
            let z4 = z4 * (-FIX_0_390_180_644);

            let z3 = z3 + z5;
            let z4 = z4 + z5;

            data[off + 7] = descale(t4 + z1 + z3, CONST_BITS - PASS1_BITS) as i16;
            data[off + 5] = descale(t5 + z2 + z4, CONST_BITS - PASS1_BITS) as i16;
            data[off + 3] = descale(t6 + z2 + z3, CONST_BITS - PASS1_BITS) as i16;
            data[off + 1] = descale(t7 + z1 + z4, CONST_BITS - PASS1_BITS) as i16;
        }

        // Pass 2: columns.
        for col in 0..8 {
            let d0 = data[col] as i32;
            let d1 = data[8 + col] as i32;
            let d2 = data[16 + col] as i32;
            let d3 = data[24 + col] as i32;
            let d4 = data[32 + col] as i32;
            let d5 = data[40 + col] as i32;
            let d6 = data[48 + col] as i32;
            let d7 = data[56 + col] as i32;

            let tmp0 = d0 + d7;
            let tmp7 = d0 - d7;
            let tmp1 = d1 + d6;
            let tmp6 = d1 - d6;
            let tmp2 = d2 + d5;
            let tmp5 = d2 - d5;
            let tmp3 = d3 + d4;
            let tmp4 = d3 - d4;

            let tmp10 = tmp0 + tmp3;
            let tmp13 = tmp0 - tmp3;
            let tmp11 = tmp1 + tmp2;
            let tmp12 = tmp1 - tmp2;

            data[col] = descale(tmp10 + tmp11, PASS1_BITS) as i16;
            data[32 + col] = descale(tmp10 - tmp11, PASS1_BITS) as i16;

            let z1 = (tmp12 + tmp13) * FIX_0_541_196_100;
            data[16 + col] =
                descale(z1 + tmp13 * FIX_0_765_366_865, CONST_BITS + PASS1_BITS) as i16;
            data[48 + col] =
                descale(z1 + tmp12 * (-FIX_1_847_759_065), CONST_BITS + PASS1_BITS) as i16;

            let z1 = tmp4 + tmp7;
            let z2 = tmp5 + tmp6;
            let z3 = tmp4 + tmp6;
            let z4 = tmp5 + tmp7;
            let z5 = (z3 + z4) * FIX_1_175_875_602;

            let t4 = tmp4 * FIX_0_298_631_336;
            let t5 = tmp5 * FIX_2_053_119_869;
            let t6 = tmp6 * FIX_3_072_711_026;
            let t7 = tmp7 * FIX_1_501_321_110;
            let z1 = z1 * (-FIX_0_899_976_223);
            let z2 = z2 * (-FIX_2_562_915_447);
            let z3 = z3 * (-FIX_1_961_570_560);
            let z4 = z4 * (-FIX_0_390_180_644);

            let z3 = z3 + z5;
            let z4 = z4 + z5;

            data[56 + col] = descale(t4 + z1 + z3, CONST_BITS + PASS1_BITS) as i16;
            data[40 + col] = descale(t5 + z2 + z4, CONST_BITS + PASS1_BITS) as i16;
            data[24 + col] = descale(t6 + z2 + z3, CONST_BITS + PASS1_BITS) as i16;
            data[8 + col] = descale(t7 + z1 + z4, CONST_BITS + PASS1_BITS) as i16;
        }
    }
}

// ===========================================================================
// quant: reciprocal-multiply quantize, natural-order output
// ===========================================================================
pub mod quant {
    use crate::quant::Divisors;

    /// Quantize a 64-element block using the precomputed divisors, in
    /// natural (DCT) order. The caller applies zig-zag separately —
    /// zig-zag is a permutation, not a hot kernel.
    ///
    /// Bit-exact equivalent to `(abs(x) + d/2) / d * sign(x)` for every
    /// `x` in `0..=2^15` — the same property libjpeg-turbo's NEON
    /// `quantize_neon` preserves.
    pub fn quantize_natural(block: &[i16; 64], div: &Divisors, out: &mut [i16; 64]) {
        for (i, &b) in block.iter().enumerate() {
            out[i] = quantize_one(b, div.recip[i], div.corr[i], div.shift[i]);
        }
    }

    #[inline(always)]
    pub(crate) fn quantize_one(temp: i16, recip: u16, corr: u16, shift: i16) -> i16 {
        let neg = temp < 0;
        let abs = temp.unsigned_abs();
        // (abs + corr) * recip, in u32 because both fit in u16.
        let product = (abs as u32 + corr as u32) * recip as u32;
        // Shift right by (shift + 16). For our typical small divisors,
        // `shift` may be negative — that's fine because our `apply_shift`
        // helper handles both directions.
        let total = shift as i32 + 16;
        let q = if total >= 0 {
            (product >> (total as u32)) as i32
        } else {
            // Negative shift means *left*; rare path (divisor==1).
            ((product as u64) << ((-total) as u32)) as i32
        };
        let q = if neg { -q } else { q };
        q.clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }
}

// ===========================================================================
// huffman: AC zero-scan helper (the only compiled SIMD-y part of huffman)
// ===========================================================================
pub mod huffman {
    /// Returns true if `block[k..k+8]` is all zero. The scalar form
    /// LLVM autovectorizes well on every backend; SIMD backends provide
    /// a directly-implemented version when the bookkeeping for a
    /// vectorized check is cheaper than autovec output.
    #[inline(always)]
    pub fn group_of_8_is_zero(block: &[i16; 64], k: usize) -> bool {
        debug_assert!(k + 8 <= 64);
        block[k..k + 8].iter().all(|&v| v == 0)
    }
}

// ===========================================================================
// Tests covering algorithmic correctness of the scalar implementations.
// Cross-check tests (NEON vs scalar etc.) live in the SIMD backends.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{RGB, RGBA};

    #[test]
    fn black_pixel_yields_zero_y() {
        let pixels = [0u8; 8 * 4];
        let mut y_row = [0u8; 8];
        let mut cb_row = [0u8; 8];
        let mut cr_row = [0u8; 8];
        color::rgb_row_to_ycc(&pixels, RGBA, 8, &mut y_row, &mut cb_row, &mut cr_row);
        for i in 0..8 {
            assert_eq!(y_row[i], 0);
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
        color::rgb_row_to_ycc(&pixels, RGBA, 8, &mut y_row, &mut cb_row, &mut cr_row);
        for i in 0..8 {
            assert_eq!(y_row[i], 255);
            assert_eq!(cb_row[i], 128);
            assert_eq!(cr_row[i], 128);
        }
    }

    #[test]
    fn rgb_layout_matches_rgba() {
        // Fixed RGB triplet — same input through bpp=3 and bpp=4 must
        // yield the same Y/Cb/Cr.
        let mut rgb = [0u8; 8 * 3];
        let mut rgba = [0u8; 8 * 4];
        for i in 0..8 {
            rgb[i * 3] = (i * 17) as u8;
            rgb[i * 3 + 1] = ((i * 23 + 7) % 256) as u8;
            rgb[i * 3 + 2] = ((i * 31 + 13) % 256) as u8;
            rgba[i * 4] = rgb[i * 3];
            rgba[i * 4 + 1] = rgb[i * 3 + 1];
            rgba[i * 4 + 2] = rgb[i * 3 + 2];
            rgba[i * 4 + 3] = 255;
        }
        let mut y3 = [0u8; 8];
        let mut cb3 = [0u8; 8];
        let mut cr3 = [0u8; 8];
        color::rgb_row_to_ycc(&rgb, RGB, 8, &mut y3, &mut cb3, &mut cr3);
        let mut y4 = [0u8; 8];
        let mut cb4 = [0u8; 8];
        let mut cr4 = [0u8; 8];
        color::rgb_row_to_ycc(&rgba, RGBA, 8, &mut y4, &mut cb4, &mut cr4);
        assert_eq!(y3, y4);
        assert_eq!(cb3, cb4);
        assert_eq!(cr3, cr4);
    }

    #[test]
    fn fdct_dc_only_block() {
        // Constant-c block ⇒ DC = 64c, all AC zero (factor of 8 vs true
        // DCT lives in the un-removed sqrt(N) per pass).
        let mut b = [50i16; 64];
        dct::fdct_islow(&mut b);
        assert_eq!(b[0], 50 * 64);
        for (i, &v) in b.iter().enumerate().skip(1) {
            assert_eq!(v, 0, "AC[{i}] not zero");
        }
    }

    #[test]
    fn quantize_reciprocal_round_trip_small() {
        use crate::quant::{Divisors, build_divisors};
        // For typical small divisors the reciprocal-multiply form must
        // match `(x + d/2) / d` for every x in our coefficient range.
        for d in [1u8, 1, 11, 99, 255] {
            let qtab = [d; 64];
            let div: Divisors = build_divisors(&qtab);
            // Every coefficient sees the same divisor here, so any index works.
            let recip = div.recip[0];
            let corr = div.corr[0];
            let shift = div.shift[0];
            let dval = (d as i32) << 3;
            for x in -8192..=8192i32 {
                let want = if dval <= 1 {
                    x
                } else {
                    let abs = x.unsigned_abs();
                    let q = ((abs + dval as u32 / 2) / dval as u32) as i32;
                    if x < 0 { -q } else { q }
                };
                let got = quant::quantize_one(x as i16, recip, corr, shift) as i32;
                if want.abs() <= i16::MAX as i32 {
                    assert_eq!(got, want, "d={d} x={x}");
                }
            }
        }
    }
}
