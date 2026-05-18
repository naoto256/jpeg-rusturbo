//! AArch64 NEON kernels — translations of libjpeg-turbo's
//! `simd/arm/jccolor-neon.c`, `jcsample-neon.c`, `jfdctint-neon.c`, and
//! `jquanti-neon.c`. See `NOTICE.md` for the upstream
//! BSD-3-Clause + IJG notice.
//!
//! Output is bit-exact identical to `arch::scalar` — the cross-check
//! tests at the bottom of this file assert that on a panel of inputs.

#![allow(dead_code)]

// ===========================================================================
// color: 16-pixel-wide RGB(A) → YCbCr, chroma downsample (4:2:0 and 4:2:2 NEON)
// ===========================================================================
pub mod color {
    use core::arch::aarch64::*;

    use crate::color::{CHROMA_BIAS, PixelLayout};

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

    /// Convert one row of `n` pixels (must be 8 or 16) to YCbCr, chroma
    /// centered on 128. Bit-exact with `arch::scalar::color::rgb_row_to_ycc`.
    pub fn rgb_row_to_ycc(
        pixels: &[u8],
        layout: PixelLayout,
        n: usize,
        y: &mut [u8],
        cb: &mut [u8],
        cr: &mut [u8],
    ) {
        debug_assert!(n == 8 || n == 16);
        debug_assert!(y.len() >= n && cb.len() >= n && cr.len() >= n);
        debug_assert!(pixels.len() >= n * layout.bpp);
        unsafe {
            if n == 16 {
                rgb_row_16_inner(
                    pixels.as_ptr(),
                    layout.bpp,
                    y.as_mut_ptr(),
                    cb.as_mut_ptr(),
                    cr.as_mut_ptr(),
                );
            } else {
                rgb_row_8_inner(
                    pixels.as_ptr(),
                    layout.bpp,
                    y.as_mut_ptr(),
                    cb.as_mut_ptr(),
                    cr.as_mut_ptr(),
                );
            }
        }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON is available, so this
    /// function is only "unsafe" in the syntactic `target_feature`
    /// sense. All inputs are by-value vector lanes; no memory access.
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

        // Cb = bias - 0.16874 R - 0.33126 G + 0.5 B.
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

    /// # Safety
    /// - `inptr` must be readable for at least `16 * bpp` bytes.
    /// - `outy`, `outcb`, `outcr` must each be writable for at least 16 bytes.
    /// - `bpp` must be 3 or 4.
    /// - `target_arch = "aarch64"` guarantees NEON is available.
    #[target_feature(enable = "neon")]
    unsafe fn rgb_row_16_inner(
        inptr: *const u8,
        bpp: usize,
        outy: *mut u8,
        outcb: *mut u8,
        outcr: *mut u8,
    ) {
        unsafe {
            let consts = vld1q_u16(CONSTS.as_ptr());
            let bias = vdupq_n_u32(CHROMA_BIAS);
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
        }
    }

    /// # Safety
    /// - `inptr` must be readable for at least `8 * bpp` bytes.
    /// - `outy`, `outcb`, `outcr` must each be writable for at least 8 bytes.
    /// - `bpp` must be 3 or 4.
    /// - `target_arch = "aarch64"` guarantees NEON is available.
    #[target_feature(enable = "neon")]
    unsafe fn rgb_row_8_inner(
        inptr: *const u8,
        bpp: usize,
        outy: *mut u8,
        outcb: *mut u8,
        outcr: *mut u8,
    ) {
        unsafe {
            let consts = vld1q_u16(CONSTS.as_ptr());
            let bias = vdupq_n_u32(CHROMA_BIAS);
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
        }
    }

    /// 16x16 → 8x8 box-average chroma downsample with libjpeg's biased
    /// rounding (alternating +1 / +2 across output columns).
    pub fn h2v2_downsample(src: &[u8; 256], dst: &mut [i16; 64]) {
        unsafe { h2v2_inner(src, dst) }
    }

    /// 16x8 → 8x8 horizontal 2:1 chroma downsample with libjpeg's
    /// `{0, 1, 0, 1, ...}` bias. Bit-exact equivalent to
    /// `arch::scalar::color::h2v1_downsample`.
    pub fn h2v1_downsample(src: &[u8; 128], dst: &mut [i16; 64]) {
        unsafe { h2v1_inner(src, dst) }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON is available. `src` /
    /// `dst` are fixed-size references.
    #[target_feature(enable = "neon")]
    unsafe fn h2v1_inner(src: &[u8; 128], dst: &mut [i16; 64]) {
        unsafe {
            // Bias `{0, 1, 0, 1, ...}` over the 8 u16 output lanes per
            // row, packed as u32 lanes (low half 0, high half 1).
            let bias = vreinterpretq_u16_u32(vdupq_n_u32(0x0001_0000));
            let level_shift = vdupq_n_s16(128);
            for j in 0..8 {
                let row = vld1q_u8(src.as_ptr().add(j * 16));
                let sums = vpadalq_u8(bias, row);
                let avg_u8 = vshrn_n_u16::<1>(sums);
                let avg_u16 = vmovl_u8(avg_u8);
                let signed = vsubq_s16(vreinterpretq_s16_u16(avg_u16), level_shift);
                vst1q_s16(dst.as_mut_ptr().add(j * 8), signed);
            }
        }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON is available. `src` /
    /// `dst` are fixed-size references — no caller-side invariants
    /// beyond the standard reference rules.
    #[target_feature(enable = "neon")]
    unsafe fn h2v2_inner(src: &[u8; 256], dst: &mut [i16; 64]) {
        unsafe {
            // Row 0 bias { 1, 0, 1, 0, ... } and row 1 bias { 0, 2, 0, 2, ... }
            // combined into one row pair = { 1, 2, 1, 2, ... } over 16-bit lanes.
            let bias = vreinterpretq_u16_u32(vdupq_n_u32(0x0002_0001));
            let level_shift = vdupq_n_s16(128);
            for j in 0..8 {
                let row0_off = j * 2 * 16;
                let row1_off = row0_off + 16;
                let r0 = vld1q_u8(src.as_ptr().add(row0_off));
                let r1 = vld1q_u8(src.as_ptr().add(row1_off));
                let sums = vpadalq_u8(bias, r0);
                let sums = vpadalq_u8(sums, r1);
                let avg_u8 = vshrn_n_u16::<2>(sums);
                let avg_u16 = vmovl_u8(avg_u8);
                let signed = vsubq_s16(vreinterpretq_s16_u16(avg_u16), level_shift);
                vst1q_s16(dst.as_mut_ptr().add(j * 8), signed);
            }
        }
    }
}

// ===========================================================================
// dct: NEON forward DCT (12-mul integer LL&M)
// ===========================================================================
pub mod dct {
    use core::arch::aarch64::*;

    const PASS1_BITS: i32 = 2;
    const DESCALE_P1: i32 = 13 - PASS1_BITS; // CONST_BITS - PASS1_BITS
    const DESCALE_P2: i32 = 13 + PASS1_BITS; // CONST_BITS + PASS1_BITS

    // Constants laid out exactly as the C `jsimd_fdct_islow_neon_consts`
    // array, indexed via `vmull_lane_s16` / `vmlal_lane_s16` in groups
    // of four 16-bit entries.
    const CONSTS: [i16; 12] = [
        2446, -3196, 4433, 6270, // F_0_298, -F_0_390,  F_0_541,  F_0_765
        -7373, 9633, 12299, -15137, // -F_0_899,  F_1_175,  F_1_501, -F_1_847
        -16069, 16819, -20995, 25172, // -F_1_961,  F_2_053, -F_2_562,  F_3_072
    ];

    /// NEON forward DCT, in-place. Bit-exact equivalent to
    /// `arch::scalar::dct::fdct_islow`.
    pub fn fdct_islow(data: &mut [i16; 64]) {
        unsafe { fdct_islow_inner(data) }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON. `data` is a
    /// fixed-size mut reference; no caller-side invariants beyond
    /// the standard reference rules.
    #[target_feature(enable = "neon")]
    unsafe fn fdct_islow_inner(data: &mut [i16; 64]) {
        unsafe {
            let consts1 = vld1_s16(CONSTS.as_ptr());
            let consts2 = vld1_s16(CONSTS.as_ptr().add(4));
            let consts3 = vld1_s16(CONSTS.as_ptr().add(8));

            // Load 8 rows, then transpose so each register holds one column.
            let s_rows_0123 = vld4q_s16(data.as_ptr());
            let s_rows_4567 = vld4q_s16(data.as_ptr().add(4 * 8));

            let cols_04 = vuzpq_s16(s_rows_0123.0, s_rows_4567.0);
            let cols_15 = vuzpq_s16(s_rows_0123.1, s_rows_4567.1);
            let cols_26 = vuzpq_s16(s_rows_0123.2, s_rows_4567.2);
            let cols_37 = vuzpq_s16(s_rows_0123.3, s_rows_4567.3);

            let mut col0 = cols_04.0;
            let mut col1 = cols_15.0;
            let mut col2 = cols_26.0;
            let mut col3 = cols_37.0;
            let mut col4 = cols_04.1;
            let mut col5 = cols_15.1;
            let mut col6 = cols_26.1;
            let mut col7 = cols_37.1;

            // -------- Pass 1: rows (registers currently hold columns). --------
            let tmp0 = vaddq_s16(col0, col7);
            let tmp7 = vsubq_s16(col0, col7);
            let tmp1 = vaddq_s16(col1, col6);
            let tmp6 = vsubq_s16(col1, col6);
            let tmp2 = vaddq_s16(col2, col5);
            let tmp5 = vsubq_s16(col2, col5);
            let tmp3 = vaddq_s16(col3, col4);
            let tmp4 = vsubq_s16(col3, col4);

            let tmp10 = vaddq_s16(tmp0, tmp3);
            let tmp13 = vsubq_s16(tmp0, tmp3);
            let tmp11 = vaddq_s16(tmp1, tmp2);
            let tmp12 = vsubq_s16(tmp1, tmp2);

            col0 = vshlq_n_s16(vaddq_s16(tmp10, tmp11), PASS1_BITS as _);
            col4 = vshlq_n_s16(vsubq_s16(tmp10, tmp11), PASS1_BITS as _);

            let tmp12_add_tmp13 = vaddq_s16(tmp12, tmp13);
            let z1_l = vmull_lane_s16::<2>(vget_low_s16(tmp12_add_tmp13), consts1);
            let z1_h = vmull_lane_s16::<2>(vget_high_s16(tmp12_add_tmp13), consts1);

            let col2_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp13), consts1);
            let col2_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp13), consts1);
            col2 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P1>(col2_l),
                vrshrn_n_s32::<DESCALE_P1>(col2_h),
            );

            let col6_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp12), consts2);
            let col6_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp12), consts2);
            col6 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P1>(col6_l),
                vrshrn_n_s32::<DESCALE_P1>(col6_h),
            );

            // Odd part.
            let z1 = vaddq_s16(tmp4, tmp7);
            let z2 = vaddq_s16(tmp5, tmp6);
            let z3 = vaddq_s16(tmp4, tmp6);
            let z4 = vaddq_s16(tmp5, tmp7);
            let mut z5_l = vmull_lane_s16::<1>(vget_low_s16(z3), consts2);
            let mut z5_h = vmull_lane_s16::<1>(vget_high_s16(z3), consts2);
            z5_l = vmlal_lane_s16::<1>(z5_l, vget_low_s16(z4), consts2);
            z5_h = vmlal_lane_s16::<1>(z5_h, vget_high_s16(z4), consts2);

            let mut tmp4_l = vmull_lane_s16::<0>(vget_low_s16(tmp4), consts1);
            let mut tmp4_h = vmull_lane_s16::<0>(vget_high_s16(tmp4), consts1);
            let mut tmp5_l = vmull_lane_s16::<1>(vget_low_s16(tmp5), consts3);
            let mut tmp5_h = vmull_lane_s16::<1>(vget_high_s16(tmp5), consts3);
            let mut tmp6_l = vmull_lane_s16::<3>(vget_low_s16(tmp6), consts3);
            let mut tmp6_h = vmull_lane_s16::<3>(vget_high_s16(tmp6), consts3);
            let mut tmp7_l = vmull_lane_s16::<2>(vget_low_s16(tmp7), consts2);
            let mut tmp7_h = vmull_lane_s16::<2>(vget_high_s16(tmp7), consts2);

            let z1_l = vmull_lane_s16::<0>(vget_low_s16(z1), consts2);
            let z1_h = vmull_lane_s16::<0>(vget_high_s16(z1), consts2);
            let z2_l = vmull_lane_s16::<2>(vget_low_s16(z2), consts3);
            let z2_h = vmull_lane_s16::<2>(vget_high_s16(z2), consts3);
            let mut z3_l = vmull_lane_s16::<0>(vget_low_s16(z3), consts3);
            let mut z3_h = vmull_lane_s16::<0>(vget_high_s16(z3), consts3);
            let mut z4_l = vmull_lane_s16::<1>(vget_low_s16(z4), consts1);
            let mut z4_h = vmull_lane_s16::<1>(vget_high_s16(z4), consts1);

            z3_l = vaddq_s32(z3_l, z5_l);
            z3_h = vaddq_s32(z3_h, z5_h);
            z4_l = vaddq_s32(z4_l, z5_l);
            z4_h = vaddq_s32(z4_h, z5_h);

            tmp4_l = vaddq_s32(tmp4_l, z1_l);
            tmp4_h = vaddq_s32(tmp4_h, z1_h);
            tmp4_l = vaddq_s32(tmp4_l, z3_l);
            tmp4_h = vaddq_s32(tmp4_h, z3_h);
            col7 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P1>(tmp4_l),
                vrshrn_n_s32::<DESCALE_P1>(tmp4_h),
            );

            tmp5_l = vaddq_s32(tmp5_l, z2_l);
            tmp5_h = vaddq_s32(tmp5_h, z2_h);
            tmp5_l = vaddq_s32(tmp5_l, z4_l);
            tmp5_h = vaddq_s32(tmp5_h, z4_h);
            col5 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P1>(tmp5_l),
                vrshrn_n_s32::<DESCALE_P1>(tmp5_h),
            );

            tmp6_l = vaddq_s32(tmp6_l, z2_l);
            tmp6_h = vaddq_s32(tmp6_h, z2_h);
            tmp6_l = vaddq_s32(tmp6_l, z3_l);
            tmp6_h = vaddq_s32(tmp6_h, z3_h);
            col3 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P1>(tmp6_l),
                vrshrn_n_s32::<DESCALE_P1>(tmp6_h),
            );

            tmp7_l = vaddq_s32(tmp7_l, z1_l);
            tmp7_h = vaddq_s32(tmp7_h, z1_h);
            tmp7_l = vaddq_s32(tmp7_l, z4_l);
            tmp7_h = vaddq_s32(tmp7_h, z4_h);
            col1 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P1>(tmp7_l),
                vrshrn_n_s32::<DESCALE_P1>(tmp7_h),
            );

            // Transpose so each register now holds a row.
            let cols_01 = vtrnq_s16(col0, col1);
            let cols_23 = vtrnq_s16(col2, col3);
            let cols_45 = vtrnq_s16(col4, col5);
            let cols_67 = vtrnq_s16(col6, col7);

            let cols_0145_l = vtrnq_s32(
                vreinterpretq_s32_s16(cols_01.0),
                vreinterpretq_s32_s16(cols_45.0),
            );
            let cols_0145_h = vtrnq_s32(
                vreinterpretq_s32_s16(cols_01.1),
                vreinterpretq_s32_s16(cols_45.1),
            );
            let cols_2367_l = vtrnq_s32(
                vreinterpretq_s32_s16(cols_23.0),
                vreinterpretq_s32_s16(cols_67.0),
            );
            let cols_2367_h = vtrnq_s32(
                vreinterpretq_s32_s16(cols_23.1),
                vreinterpretq_s32_s16(cols_67.1),
            );

            let rows_04 = vzipq_s32(cols_0145_l.0, cols_2367_l.0);
            let rows_15 = vzipq_s32(cols_0145_h.0, cols_2367_h.0);
            let rows_26 = vzipq_s32(cols_0145_l.1, cols_2367_l.1);
            let rows_37 = vzipq_s32(cols_0145_h.1, cols_2367_h.1);

            let mut row0 = vreinterpretq_s16_s32(rows_04.0);
            let mut row1 = vreinterpretq_s16_s32(rows_15.0);
            let mut row2 = vreinterpretq_s16_s32(rows_26.0);
            let mut row3 = vreinterpretq_s16_s32(rows_37.0);
            let mut row4 = vreinterpretq_s16_s32(rows_04.1);
            let mut row5 = vreinterpretq_s16_s32(rows_15.1);
            let mut row6 = vreinterpretq_s16_s32(rows_26.1);
            let mut row7 = vreinterpretq_s16_s32(rows_37.1);

            // -------- Pass 2: columns. --------
            let tmp0 = vaddq_s16(row0, row7);
            let tmp7 = vsubq_s16(row0, row7);
            let tmp1 = vaddq_s16(row1, row6);
            let tmp6 = vsubq_s16(row1, row6);
            let tmp2 = vaddq_s16(row2, row5);
            let tmp5 = vsubq_s16(row2, row5);
            let tmp3 = vaddq_s16(row3, row4);
            let tmp4 = vsubq_s16(row3, row4);

            let tmp10 = vaddq_s16(tmp0, tmp3);
            let tmp13 = vsubq_s16(tmp0, tmp3);
            let tmp11 = vaddq_s16(tmp1, tmp2);
            let tmp12 = vsubq_s16(tmp1, tmp2);

            row0 = vrshrq_n_s16::<PASS1_BITS>(vaddq_s16(tmp10, tmp11));
            row4 = vrshrq_n_s16::<PASS1_BITS>(vsubq_s16(tmp10, tmp11));

            let tmp12_add_tmp13 = vaddq_s16(tmp12, tmp13);
            let z1_l = vmull_lane_s16::<2>(vget_low_s16(tmp12_add_tmp13), consts1);
            let z1_h = vmull_lane_s16::<2>(vget_high_s16(tmp12_add_tmp13), consts1);

            let row2_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp13), consts1);
            let row2_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp13), consts1);
            row2 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P2>(row2_l),
                vrshrn_n_s32::<DESCALE_P2>(row2_h),
            );

            let row6_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp12), consts2);
            let row6_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp12), consts2);
            row6 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P2>(row6_l),
                vrshrn_n_s32::<DESCALE_P2>(row6_h),
            );

            let z1 = vaddq_s16(tmp4, tmp7);
            let z2 = vaddq_s16(tmp5, tmp6);
            let z3 = vaddq_s16(tmp4, tmp6);
            let z4 = vaddq_s16(tmp5, tmp7);

            let mut z5_l = vmull_lane_s16::<1>(vget_low_s16(z3), consts2);
            let mut z5_h = vmull_lane_s16::<1>(vget_high_s16(z3), consts2);
            z5_l = vmlal_lane_s16::<1>(z5_l, vget_low_s16(z4), consts2);
            z5_h = vmlal_lane_s16::<1>(z5_h, vget_high_s16(z4), consts2);

            let mut tmp4_l = vmull_lane_s16::<0>(vget_low_s16(tmp4), consts1);
            let mut tmp4_h = vmull_lane_s16::<0>(vget_high_s16(tmp4), consts1);
            let mut tmp5_l = vmull_lane_s16::<1>(vget_low_s16(tmp5), consts3);
            let mut tmp5_h = vmull_lane_s16::<1>(vget_high_s16(tmp5), consts3);
            let mut tmp6_l = vmull_lane_s16::<3>(vget_low_s16(tmp6), consts3);
            let mut tmp6_h = vmull_lane_s16::<3>(vget_high_s16(tmp6), consts3);
            let mut tmp7_l = vmull_lane_s16::<2>(vget_low_s16(tmp7), consts2);
            let mut tmp7_h = vmull_lane_s16::<2>(vget_high_s16(tmp7), consts2);

            let z1_l = vmull_lane_s16::<0>(vget_low_s16(z1), consts2);
            let z1_h = vmull_lane_s16::<0>(vget_high_s16(z1), consts2);
            let z2_l = vmull_lane_s16::<2>(vget_low_s16(z2), consts3);
            let z2_h = vmull_lane_s16::<2>(vget_high_s16(z2), consts3);
            let mut z3_l = vmull_lane_s16::<0>(vget_low_s16(z3), consts3);
            let mut z3_h = vmull_lane_s16::<0>(vget_high_s16(z3), consts3);
            let mut z4_l = vmull_lane_s16::<1>(vget_low_s16(z4), consts1);
            let mut z4_h = vmull_lane_s16::<1>(vget_high_s16(z4), consts1);

            z3_l = vaddq_s32(z3_l, z5_l);
            z3_h = vaddq_s32(z3_h, z5_h);
            z4_l = vaddq_s32(z4_l, z5_l);
            z4_h = vaddq_s32(z4_h, z5_h);

            tmp4_l = vaddq_s32(tmp4_l, z1_l);
            tmp4_h = vaddq_s32(tmp4_h, z1_h);
            tmp4_l = vaddq_s32(tmp4_l, z3_l);
            tmp4_h = vaddq_s32(tmp4_h, z3_h);
            row7 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P2>(tmp4_l),
                vrshrn_n_s32::<DESCALE_P2>(tmp4_h),
            );

            tmp5_l = vaddq_s32(tmp5_l, z2_l);
            tmp5_h = vaddq_s32(tmp5_h, z2_h);
            tmp5_l = vaddq_s32(tmp5_l, z4_l);
            tmp5_h = vaddq_s32(tmp5_h, z4_h);
            row5 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P2>(tmp5_l),
                vrshrn_n_s32::<DESCALE_P2>(tmp5_h),
            );

            tmp6_l = vaddq_s32(tmp6_l, z2_l);
            tmp6_h = vaddq_s32(tmp6_h, z2_h);
            tmp6_l = vaddq_s32(tmp6_l, z3_l);
            tmp6_h = vaddq_s32(tmp6_h, z3_h);
            row3 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P2>(tmp6_l),
                vrshrn_n_s32::<DESCALE_P2>(tmp6_h),
            );

            tmp7_l = vaddq_s32(tmp7_l, z1_l);
            tmp7_h = vaddq_s32(tmp7_h, z1_h);
            tmp7_l = vaddq_s32(tmp7_l, z4_l);
            tmp7_h = vaddq_s32(tmp7_h, z4_h);
            row1 = vcombine_s16(
                vrshrn_n_s32::<DESCALE_P2>(tmp7_l),
                vrshrn_n_s32::<DESCALE_P2>(tmp7_h),
            );

            let p = data.as_mut_ptr();
            vst1q_s16(p, row0);
            vst1q_s16(p.add(8), row1);
            vst1q_s16(p.add(16), row2);
            vst1q_s16(p.add(24), row3);
            vst1q_s16(p.add(32), row4);
            vst1q_s16(p.add(40), row5);
            vst1q_s16(p.add(48), row6);
            vst1q_s16(p.add(56), row7);
        }
    }
}

// ===========================================================================
// quant: NEON quantize, natural-order output
// ===========================================================================
pub mod quant {
    use core::arch::aarch64::*;

    use crate::quant::Divisors;

    /// Quantize a 64-element block using the precomputed divisors, in
    /// natural (DCT) order. Bit-exact equivalent to
    /// `arch::scalar::quant::quantize_natural`.
    pub fn quantize_natural(block: &[i16; 64], div: &Divisors, out: &mut [i16; 64]) {
        unsafe { quantize_inner(block, div, out) }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON. All inputs are
    /// fixed-size references; no caller-side invariants beyond the
    /// standard reference rules.
    #[target_feature(enable = "neon")]
    unsafe fn quantize_inner(workspace: &[i16; 64], div: &Divisors, out: &mut [i16; 64]) {
        unsafe {
            let ws = workspace.as_ptr();
            let recipp = div.recip.as_ptr();
            let corrp = div.corr.as_ptr();
            let shiftp = div.shift.as_ptr();
            let outp = out.as_mut_ptr();

            // Process 8 rows in two batches of 4 (matches the C unroll).
            let mut i = 0usize;
            while i < 8 {
                for k in 0..4usize {
                    let row_off = (i + k) * 8;
                    let row = vld1q_s16(ws.add(row_off));
                    let recip = vld1q_u16(recipp.add(row_off));
                    let corr = vld1q_u16(corrp.add(row_off));
                    let shift = vld1q_s16(shiftp.add(row_off));

                    // Sign-extract: -1 for negative lanes, 0 otherwise.
                    let sign = vshrq_n_s16::<15>(row);
                    // Absolute value, reinterpreted as u16.
                    let absv = vreinterpretq_u16_s16(vabsq_s16(row));
                    let biased = vaddq_u16(absv, corr);

                    // 16x16 → 32 multiply, then narrow back via shrn(16) to
                    // pull the high half. This is libjpeg's SIMD pattern.
                    let prod_l = vmull_u16(vget_low_u16(biased), vget_low_u16(recip));
                    let prod_h = vmull_u16(vget_high_u16(biased), vget_high_u16(recip));
                    let high16 = vcombine_s16(
                        vshrn_n_s32::<16>(vreinterpretq_s32_u32(prod_l)),
                        vshrn_n_s32::<16>(vreinterpretq_s32_u32(prod_h)),
                    );

                    // Variable right shift (shift values are >= 0 in
                    // practice; they encode "additional" shifts beyond the
                    // 16 we just did via shrn). NEON only has variable
                    // *left* shift, so negate.
                    let shifted = vreinterpretq_s16_u16(vshlq_u16(
                        vreinterpretq_u16_s16(high16),
                        vnegq_s16(shift),
                    ));

                    // Restore sign: XOR with sign mask, then subtract sign
                    // mask. (For negative lanes: ~q + 1 = -q.)
                    let signed = vsubq_s16(veorq_s16(shifted, sign), sign);
                    vst1q_s16(outp.add(row_off), signed);
                }
                i += 4;
            }
        }
    }
}

// ===========================================================================
// huffman: AC zero-scan over 8 contiguous coefficients
// ===========================================================================
pub mod huffman {
    use core::arch::aarch64::{vabsq_s16, vld1q_s16, vmaxvq_u16, vreinterpretq_u16_s16};

    /// True iff `block[k..k+8]` is all zero. Loads 8 i16 lanes, takes
    /// `vmaxvq_u16(abs)`, returns `max == 0`.
    #[inline(always)]
    pub fn group_of_8_is_zero(block: &[i16; 64], k: usize) -> bool {
        debug_assert!(k + 8 <= 64);
        // SAFETY: k + 8 <= 64; NEON intrinsics are unconditionally
        // available on aarch64.
        unsafe {
            let v = vld1q_s16(block.as_ptr().add(k));
            let abs = vabsq_s16(v);
            let max = vmaxvq_u16(vreinterpretq_u16_s16(abs));
            max == 0
        }
    }
}

// ===========================================================================
// Cross-check tests: NEON ↔ scalar bit-exact equivalence on a panel of
// inputs. Compiled only on aarch64 (where both backends are reachable).
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::scalar;
    use crate::color::RGBA;

    fn random_block(seed: u64) -> [i16; 64] {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut b = [0i16; 64];
        for v in &mut b {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 33) as i32 % 256 - 128) as i16;
        }
        b
    }

    #[test]
    fn color_neon_matches_scalar_row16() {
        // Deterministic gradient + alternating pattern.
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
        scalar::color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_s, &mut cb_s, &mut cr_s);

        let mut y_n = [0u8; 16];
        let mut cb_n = [0u8; 16];
        let mut cr_n = [0u8; 16];
        color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_n, &mut cb_n, &mut cr_n);

        assert_eq!(y_s, y_n);
        assert_eq!(cb_s, cb_n);
        assert_eq!(cr_s, cr_n);
    }

    #[test]
    fn color_neon_matches_scalar_downsample() {
        let mut src = [0u8; 256];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i * 53 + 17) % 256) as u8;
        }
        let mut a = [0i16; 64];
        let mut b = [0i16; 64];
        scalar::color::h2v2_downsample(&src, &mut a);
        color::h2v2_downsample(&src, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn color_neon_matches_scalar_h2v1_downsample() {
        let mut src = [0u8; 128];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i * 71 + 23) % 256) as u8;
        }
        let mut a = [0i16; 64];
        let mut b = [0i16; 64];
        scalar::color::h2v1_downsample(&src, &mut a);
        color::h2v1_downsample(&src, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn fdct_neon_matches_scalar_zeros() {
        let mut a = [0i16; 64];
        let mut b = [0i16; 64];
        scalar::dct::fdct_islow(&mut a);
        dct::fdct_islow(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn fdct_neon_matches_scalar_const() {
        let mut a = [42i16; 64];
        let mut b = [42i16; 64];
        scalar::dct::fdct_islow(&mut a);
        dct::fdct_islow(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn fdct_neon_matches_scalar_ramp() {
        let mut a = [0i16; 64];
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as i16) - 32;
        }
        let mut b = a;
        scalar::dct::fdct_islow(&mut a);
        dct::fdct_islow(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn fdct_neon_matches_scalar_random() {
        for seed in 0..5u64 {
            let mut a = random_block(seed);
            let mut b = a;
            scalar::dct::fdct_islow(&mut a);
            dct::fdct_islow(&mut b);
            assert_eq!(a, b, "seed={seed}");
        }
    }

    #[test]
    fn fdct_neon_matches_scalar_extremes() {
        let mut a = [0i16; 64];
        for (i, v) in a.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 127 } else { -128 };
        }
        let mut b = a;
        scalar::dct::fdct_islow(&mut a);
        dct::fdct_islow(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn quant_neon_matches_scalar() {
        use crate::quant::build_divisors;
        use crate::tables::{STD_LUMA_QUANT, scale_quant_table};

        let mut block = [0i16; 64];
        for (i, v) in block.iter_mut().enumerate() {
            // Vary sign and magnitude.
            let m = (i as i32 * 37) % 4001 - 2000;
            *v = m as i16;
        }
        let qtab = scale_quant_table(&STD_LUMA_QUANT, 80);
        let div = build_divisors(&qtab);

        let mut sout = [0i16; 64];
        let mut nout = [0i16; 64];
        scalar::quant::quantize_natural(&block, &div, &mut sout);
        quant::quantize_natural(&block, &div, &mut nout);
        assert_eq!(sout, nout);
    }

    #[test]
    fn huffman_zero_scan_matches_scalar() {
        let mut block = [0i16; 64];
        // All-zero panel.
        for k in (0..64).step_by(8) {
            assert_eq!(
                scalar::huffman::group_of_8_is_zero(&block, k),
                huffman::group_of_8_is_zero(&block, k),
                "all-zero k={k}"
            );
        }
        // Sparse non-zeros at various positions.
        block[1] = 7;
        block[15] = -3;
        block[33] = 1;
        for k in (0..=56).step_by(1) {
            assert_eq!(
                scalar::huffman::group_of_8_is_zero(&block, k),
                huffman::group_of_8_is_zero(&block, k),
                "sparse k={k}"
            );
        }
    }
}
