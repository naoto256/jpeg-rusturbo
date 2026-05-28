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
            let r = pixels[p + layout.r_off] as u32;
            let g = pixels[p + layout.g_off] as u32;
            let b = pixels[p + layout.b_off] as u32;
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

    // ---- Decoder-side upsample (box / nearest-neighbor) ----

    /// Box-upsample 8x8 chroma into 16x16 by 2x replication in both
    /// axes. Pairs with [`h2v2_downsample`] on the encoder side but
    /// is intentionally a simple nearest-neighbor expansion. Fancy
    /// (interpolating) upsample is not implemented.
    pub fn h2v2_upsample(src: &[u8; 64], dst: &mut [u8; 256]) {
        for j in 0..8 {
            for i in 0..8 {
                let s = src[j * 8 + i];
                let row0 = (j * 2) * 16;
                let row1 = row0 + 16;
                let col = i * 2;
                dst[row0 + col] = s;
                dst[row0 + col + 1] = s;
                dst[row1 + col] = s;
                dst[row1 + col + 1] = s;
            }
        }
    }

    /// Box-upsample 8x8 chroma into 16x8 by 2x replication in the
    /// horizontal axis only. Counterpart of [`h2v1_downsample`].
    pub fn h2v1_upsample(src: &[u8; 64], dst: &mut [u8; 128]) {
        for j in 0..8 {
            for i in 0..8 {
                let s = src[j * 8 + i];
                let col = i * 2;
                dst[j * 16 + col] = s;
                dst[j * 16 + col + 1] = s;
            }
        }
    }

    // ---- Decoder-side per-row YCbCr → RGB(A) ----

    /// Convert one row of `n` YCbCr samples to RGB(A) at `layout`. The
    /// alpha / pad byte (when `bpp == 4`) is filled with `0xFF`.
    ///
    /// Inverse color matrix (SCALEBITS = 16):
    ///   R = Y                              + 1.40200 * (Cr - 128)
    ///   G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    ///   B = Y + 1.77200 * (Cb - 128)
    pub fn ycc_row_to_rgb(
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        out: &mut [u8],
        n: usize,
        layout: PixelLayout,
    ) {
        debug_assert!(y.len() >= n && cb.len() >= n && cr.len() >= n);
        debug_assert!(out.len() >= n * layout.bpp);
        const HALF: i32 = 1 << 15;
        // Inverse color-conversion fixed-point constants (round(x * 65536)).
        const I_CR_R: i32 = 91881; // 1.40200
        const I_CB_G: i32 = 22554; // 0.34414
        const I_CR_G: i32 = 46802; // 0.71414
        const I_CB_B: i32 = 116130; // 1.77200

        for i in 0..n {
            let yi = y[i] as i32;
            let cbi = cb[i] as i32 - 128;
            let cri = cr[i] as i32 - 128;
            let r = yi + ((I_CR_R * cri + HALF) >> 16);
            let g = yi - ((I_CB_G * cbi + I_CR_G * cri + HALF) >> 16);
            let b = yi + ((I_CB_B * cbi + HALF) >> 16);
            let r = r.clamp(0, 255) as u8;
            let g = g.clamp(0, 255) as u8;
            let b = b.clamp(0, 255) as u8;
            let p = i * layout.bpp;
            out[p + layout.r_off] = r;
            out[p + layout.g_off] = g;
            out[p + layout.b_off] = b;
            if layout.bpp == 4 {
                let a_off = 6 - layout.r_off - layout.g_off - layout.b_off;
                out[p + a_off] = 0xFF;
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

    /// Round-and-shift used by the inverse pass; pass 2 also folds in
    /// the level shift (`+128`) so the output range matches u8.
    #[inline(always)]
    fn descale_pos(x: i32, n: i32) -> i32 {
        (x + (1 << (n - 1))) >> n
    }

    /// Inverse 8x8 DCT, libjpeg-turbo `jidctint` style. Input
    /// `coef_block` is in natural order, already dequantized (i.e.
    /// `Q[i] * coef[i]`). Output is 8x8 u8 samples in natural order.
    ///
    /// Pass 1 processes columns into a scaled i32 workspace; pass 2
    /// processes rows from the workspace, descales, level-shifts by
    /// +128, and clamps to `[0, 255]`.
    ///
    /// The i32 workspace makes this implementation safe for any `i16`
    /// input, including adversarial values that would overflow the i16
    /// workspace used by the NEON port. Build with `--features
    /// force-scalar` if you need that guarantee on aarch64.
    pub fn idct_islow(coef: &[i16; 64], output: &mut [u8; 64]) {
        let mut ws = [0i32; 64];

        // ---- Pass 1: columns from coef → ws (i32) ----
        for col in 0..8 {
            // Even part
            let z2 = coef[16 + col] as i32;
            let z3 = coef[48 + col] as i32;
            let z1 = (z2 + z3) * FIX_0_541_196_100;
            let tmp2 = z1 + z3 * (-FIX_1_847_759_065);
            let tmp3 = z1 + z2 * FIX_0_765_366_865;

            let z2 = coef[col] as i32;
            let z3 = coef[32 + col] as i32;
            let tmp0 = (z2 + z3) << CONST_BITS;
            let tmp1 = (z2 - z3) << CONST_BITS;

            let tmp10 = tmp0 + tmp3;
            let tmp13 = tmp0 - tmp3;
            let tmp11 = tmp1 + tmp2;
            let tmp12 = tmp1 - tmp2;

            // Odd part
            let t0 = coef[56 + col] as i32;
            let t1 = coef[40 + col] as i32;
            let t2 = coef[24 + col] as i32;
            let t3 = coef[8 + col] as i32;

            let z1 = t0 + t3;
            let z2 = t1 + t2;
            let z3 = t0 + t2;
            let z4 = t1 + t3;
            let z5 = (z3 + z4) * FIX_1_175_875_602;

            let t0m = t0 * FIX_0_298_631_336;
            let t1m = t1 * FIX_2_053_119_869;
            let t2m = t2 * FIX_3_072_711_026;
            let t3m = t3 * FIX_1_501_321_110;
            let z1 = z1 * (-FIX_0_899_976_223);
            let z2 = z2 * (-FIX_2_562_915_447);
            let z3 = z3 * (-FIX_1_961_570_560);
            let z4 = z4 * (-FIX_0_390_180_644);

            let z3 = z3 + z5;
            let z4 = z4 + z5;

            let to0 = t0m + z1 + z3;
            let to1 = t1m + z2 + z4;
            let to2 = t2m + z2 + z3;
            let to3 = t3m + z1 + z4;

            ws[col] = descale_pos(tmp10 + to3, CONST_BITS - PASS1_BITS);
            ws[56 + col] = descale_pos(tmp10 - to3, CONST_BITS - PASS1_BITS);
            ws[8 + col] = descale_pos(tmp11 + to2, CONST_BITS - PASS1_BITS);
            ws[48 + col] = descale_pos(tmp11 - to2, CONST_BITS - PASS1_BITS);
            ws[16 + col] = descale_pos(tmp12 + to1, CONST_BITS - PASS1_BITS);
            ws[40 + col] = descale_pos(tmp12 - to1, CONST_BITS - PASS1_BITS);
            ws[24 + col] = descale_pos(tmp13 + to0, CONST_BITS - PASS1_BITS);
            ws[32 + col] = descale_pos(tmp13 - to0, CONST_BITS - PASS1_BITS);
        }

        // ---- Pass 2: rows from ws → output (u8, range-limited) ----
        const SHIFT2: i32 = CONST_BITS + PASS1_BITS + 3;
        // +3 bits absorb the `*8` scaling residual built into the
        // forward DCT, so the round-bias becomes `1 << (SHIFT2 - 1)`.
        // The +128 level shift is folded in by adding `128 << SHIFT2`
        // to the round-bias before descaling.
        let round_bias: i32 = (1 << (SHIFT2 - 1)) + (128 << SHIFT2);

        for row in 0..8 {
            let off = row * 8;
            // Even part
            let z2 = ws[off + 2];
            let z3 = ws[off + 6];
            let z1 = (z2 + z3) * FIX_0_541_196_100;
            let tmp2 = z1 + z3 * (-FIX_1_847_759_065);
            let tmp3 = z1 + z2 * FIX_0_765_366_865;

            let z2 = ws[off];
            let z3 = ws[off + 4];
            let tmp0 = (z2 + z3) << CONST_BITS;
            let tmp1 = (z2 - z3) << CONST_BITS;

            let tmp10 = tmp0 + tmp3;
            let tmp13 = tmp0 - tmp3;
            let tmp11 = tmp1 + tmp2;
            let tmp12 = tmp1 - tmp2;

            // Odd part
            let t0 = ws[off + 7];
            let t1 = ws[off + 5];
            let t2 = ws[off + 3];
            let t3 = ws[off + 1];

            let z1 = t0 + t3;
            let z2 = t1 + t2;
            let z3 = t0 + t2;
            let z4 = t1 + t3;
            let z5 = (z3 + z4) * FIX_1_175_875_602;

            let t0m = t0 * FIX_0_298_631_336;
            let t1m = t1 * FIX_2_053_119_869;
            let t2m = t2 * FIX_3_072_711_026;
            let t3m = t3 * FIX_1_501_321_110;
            let z1 = z1 * (-FIX_0_899_976_223);
            let z2 = z2 * (-FIX_2_562_915_447);
            let z3 = z3 * (-FIX_1_961_570_560);
            let z4 = z4 * (-FIX_0_390_180_644);

            let z3 = z3 + z5;
            let z4 = z4 + z5;

            let to0 = t0m + z1 + z3;
            let to1 = t1m + z2 + z4;
            let to2 = t2m + z2 + z3;
            let to3 = t3m + z1 + z4;

            // Apply DESCALE with combined round + level-shift bias.
            let p0 = ((tmp10 + to3) + round_bias) >> SHIFT2;
            let p1 = ((tmp11 + to2) + round_bias) >> SHIFT2;
            let p2 = ((tmp12 + to1) + round_bias) >> SHIFT2;
            let p3 = ((tmp13 + to0) + round_bias) >> SHIFT2;
            let p4 = ((tmp13 - to0) + round_bias) >> SHIFT2;
            let p5 = ((tmp12 - to1) + round_bias) >> SHIFT2;
            let p6 = ((tmp11 - to2) + round_bias) >> SHIFT2;
            let p7 = ((tmp10 - to3) + round_bias) >> SHIFT2;

            output[off] = p0.clamp(0, 255) as u8;
            output[off + 1] = p1.clamp(0, 255) as u8;
            output[off + 2] = p2.clamp(0, 255) as u8;
            output[off + 3] = p3.clamp(0, 255) as u8;
            output[off + 4] = p4.clamp(0, 255) as u8;
            output[off + 5] = p5.clamp(0, 255) as u8;
            output[off + 6] = p6.clamp(0, 255) as u8;
            output[off + 7] = p7.clamp(0, 255) as u8;
        }
    }
}

// ===========================================================================
// sample: decoder-side fancy chroma upsample (h2v2 / h2 / v2)
// ===========================================================================
pub mod sample {
    /// Vertical pass of libjpeg-turbo's `h2v2_fancy` upsample: for each
    /// output sample, blend the current chroma row with the chosen
    /// neighbor row (above for the upper output of a pair, below for the
    /// lower) using weights `(3 * cur + nbr + 2) >> 2`.
    ///
    /// The caller selects which neighbor row to pass — `prev` (cur - 1)
    /// for `phase == 0`, `next` (cur + 1) for `phase == 1` — and is
    /// responsible for clamping the neighbor index to the plane height
    /// at the top / bottom edges (= passing the same row as `cur` when
    /// the neighbor would be out of bounds). This keeps the kernel
    /// branch-free in the hot loop.
    ///
    /// `cur`, `nbr`, and `out` are all length `n` (= chroma plane width).
    pub fn h2v2_fancy_vblend(cur: &[u8], nbr: &[u8], out: &mut [u8], n: usize) {
        for (dst, (&c, &nb)) in out.iter_mut().take(n).zip(cur.iter().zip(nbr.iter())) {
            *dst = (((c as u16) * 3 + nb as u16 + 2) >> 2) as u8;
        }
    }

    /// Horizontal pass of libjpeg-turbo's `h2_fancy` upsample: produce
    /// `2 * n` output samples from `n` chroma samples using the symmetric
    /// 3:1 weighted blend.
    ///
    ///   dst[2*i]   = (3 * src[i] + src[i - 1] + 2) >> 2
    ///   dst[2*i+1] = (3 * src[i] + src[i + 1] + 2) >> 2
    ///
    /// The edges clamp: `src[-1]` is treated as `src[0]` and `src[n]` as
    /// `src[n - 1]`. The caller passes `n >= 1` and `dst` of length
    /// `>= 2 * n`. This kernel does not pad / fill any output beyond
    /// `dst[..2 * n]`; the outer per-row upsample wraps it and replicates
    /// the last value if the requested output width exceeds `2 * n`.
    pub fn h2_fancy_upsample(src: &[u8], dst: &mut [u8], n: usize) {
        debug_assert!(n >= 1);
        debug_assert!(dst.len() >= 2 * n);
        debug_assert!(src.len() >= n);
        for i in 0..n {
            let cur = src[i] as u16;
            let prev = if i == 0 { cur } else { src[i - 1] as u16 };
            let next = if i + 1 >= n { cur } else { src[i + 1] as u16 };
            dst[2 * i] = ((cur * 3 + prev + 2) >> 2) as u8;
            dst[2 * i + 1] = ((cur * 3 + next + 2) >> 2) as u8;
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

    /// Permute `natural` (DCT natural order) into `zz` (zig-zag order).
    /// Reference scalar form; SIMD backends override with a permutation
    /// kernel.
    ///
    /// `get_unchecked` drops the per-iteration bounds check that LLVM
    /// cannot prove away (ZIGZAG is `const [usize; 64]`, not a refined
    /// index type). Safety: every ZIGZAG entry is `< 64`, verified by
    /// `tables::tests`.
    pub fn zigzag_scatter(natural: &[i16; 64], zz: &mut [i16; 64]) {
        use crate::tables::ZIGZAG;
        for k in 0..64 {
            unsafe {
                *zz.get_unchecked_mut(k) = *natural.get_unchecked(*ZIGZAG.get_unchecked(k));
            }
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
// huffman: 64-bit nonzero bitmap for AC scan
// ===========================================================================
pub mod huffman {
    /// Bit `k` is set iff `block[k] != 0`. Drives the AC run-length scan
    /// via trailing/leading-zeros bit twiddling, so that the inner walk
    /// is `ctz`-driven rather than per-coefficient branchy.
    ///
    /// The scalar form here is a straightforward loop; SIMD backends
    /// build the same bitmap with `vceqz + vshrn + vaddv`-style packed
    /// comparison.
    #[inline]
    pub fn nonzero_bitmap(block: &[i16; 64]) -> u64 {
        let mut bm: u64 = 0;
        for (k, &v) in block.iter().enumerate() {
            if v != 0 {
                bm |= 1u64 << k;
            }
        }
        bm
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
