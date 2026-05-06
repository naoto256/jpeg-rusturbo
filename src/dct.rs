//! Forward 8x8 DCT — integer LL&M (Loeffler/Ligtenberg/Moschytz)
//! "slow / accurate" variant, a.k.a. `jpeg_fdct_islow`.
//!
//! This is the algorithm libjpeg-turbo uses by default: 12 multiplies +
//! 32 adds per 1-D pass, no data path with more than one multiply,
//! 13-bit fixed-point constants. The output is a 16-bit DCT coefficient
//! that is *scaled by 8* relative to a true DCT — that factor of 8 is
//! folded into the quantization step (see `quant.rs`).
//!
//! Phase 2 plumbing:
//!  - The scalar implementation here is the canonical reference.
//!  - The NEON path (`fdct_islow_neon`, gated on `aarch64`) translates
//!    libjpeg-turbo's `simd/arm/jfdctint-neon.c` and is unit-tested
//!    bit-exact against the scalar version.
//!
//! Algorithmic notes mirror `jfdctint.c` so that the structure is
//! verifiable line-for-line. Variable names follow the C source.

// On aarch64 the scalar reference is only reached from tests (the
// dispatcher in `lib.rs` always calls the NEON kernel). Suppress the
// "unused" warnings the scalar fallback would otherwise emit.
#![allow(dead_code)]

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

/// Forward 8x8 DCT (integer LL&M, scalar reference). Operates in place
/// on a 64-element block of i16 samples in row-major order. Inputs are
/// expected to be already level-shifted (Y in `[-128, 127]`, Cb/Cr also
/// centered on 0). Outputs are 16-bit DCT coefficients scaled by 8.
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

        // Even part.
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[off] = ((tmp10 + tmp11) << PASS1_BITS) as i16;
        data[off + 4] = ((tmp10 - tmp11) << PASS1_BITS) as i16;

        let z1 = (tmp12 + tmp13) * FIX_0_541_196_100;
        data[off + 2] =
            descale(z1 + tmp13 * FIX_0_765_366_865, CONST_BITS - PASS1_BITS) as i16;
        data[off + 6] =
            descale(z1 + tmp12 * (-FIX_1_847_759_065), CONST_BITS - PASS1_BITS) as i16;

        // Odd part.
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

    // Pass 2: columns. Removes the PASS1_BITS scaling and leaves an
    // overall factor of 8 (the un-removed sqrt(8) per pass).
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

// ===========================================================================
// AArch64 NEON kernel — translation of libjpeg-turbo's
// `simd/arm/jfdctint-neon.c`. See `LICENSES/libjpeg-turbo.txt` for the
// upstream zlib-style notice; nothing in the original required source-level
// attribution, but we keep the structure parallel for review.
// ===========================================================================
#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use neon::fdct_islow_neon;

#[cfg(target_arch = "aarch64")]
mod neon {
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

    /// NEON forward DCT, in-place. Bit-exact equivalent to `fdct_islow`.
    ///
    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON; the function is safe.
    pub fn fdct_islow_neon(data: &mut [i16; 64]) {
        unsafe { fdct_islow_neon_inner(data) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn fdct_islow_neon_inner(data: &mut [i16; 64]) {
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

        // Even part.
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
        // sqrt(2) * c3
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_block() -> [i16; 64] {
        let mut b = [0i16; 64];
        for (i, v) in b.iter_mut().enumerate() {
            *v = (i as i16) - 32;
        }
        b
    }

    fn random_block(seed: u64) -> [i16; 64] {
        // Tiny LCG, deterministic.
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut b = [0i16; 64];
        for v in &mut b {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to roughly the legal level-shifted sample range.
            *v = ((s >> 33) as i32 % 256 - 128) as i16;
        }
        b
    }

    #[test]
    fn dc_only_block() {
        // A solid block: the DCT puts all energy in DC.
        //
        // For an 8x8 block of constant c the LL&M islow output at DC
        // is 64c (per-pass row sum is 8c, the second pass sums 8 rows
        // and shifts down by PASS1_BITS=2). The factor of 8 vs the
        // "true" DCT is the un-removed sqrt(N) per pass that the
        // quantizer divides out via `quant_table << 3` (see quant.rs).
        let mut b = [50i16; 64];
        fdct_islow(&mut b);
        assert_eq!(b[0], 50 * 64);
        for (i, &v) in b.iter().enumerate().skip(1) {
            assert_eq!(v, 0, "AC[{i}] not zero");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_zeros() {
        let mut a = [0i16; 64];
        let mut b = [0i16; 64];
        fdct_islow(&mut a);
        neon::fdct_islow_neon(&mut b);
        assert_eq!(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_const() {
        let mut a = [42i16; 64];
        let mut b = [42i16; 64];
        fdct_islow(&mut a);
        neon::fdct_islow_neon(&mut b);
        assert_eq!(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_ramp() {
        let mut a = ramp_block();
        let mut b = a;
        fdct_islow(&mut a);
        neon::fdct_islow_neon(&mut b);
        assert_eq!(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_random() {
        for seed in 0..5u64 {
            let mut a = random_block(seed);
            let mut b = a;
            fdct_islow(&mut a);
            neon::fdct_islow_neon(&mut b);
            assert_eq!(a, b, "seed={seed}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_extremes() {
        // Worst-case input range for level-shifted 8-bit: [-128, 127].
        let mut a = [0i16; 64];
        for (i, v) in a.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 127 } else { -128 };
        }
        let mut b = a;
        fdct_islow(&mut a);
        neon::fdct_islow_neon(&mut b);
        assert_eq!(a, b);
    }
}
