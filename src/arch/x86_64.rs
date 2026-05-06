//! x86_64 SIMD kernels — translations of libjpeg-turbo's
//! `simd/x86_64/*-avx2.asm`. See `LICENSES/libjpeg-turbo.txt`.
//!
//! Backend status (incremental port):
//!   - `quant`    — AVX2
//!   - `dct`      — AVX2
//!   - `color`    — delegate to scalar (TODO)
//!   - `huffman`  — delegate to scalar (TODO)
//!
//! Runtime feature detection: AVX2 is the only target we ever dispatch
//! to from x86_64. Non-AVX2 CPUs hit the scalar fallback at runtime via
//! `is_x86_feature_detected!`. The result is cached after first call.

#![allow(dead_code)]

// Kernels not yet ported: forward to the scalar reference.
pub mod color {
    pub use crate::arch::scalar::color::*;
}
pub mod huffman {
    pub use crate::arch::scalar::huffman::*;
}

// ===========================================================================
// quant: AVX2 reciprocal-multiply quantize, natural-order output.
// Translated from `simd/x86_64/jquanti-avx2.asm::jsimd_quantize_avx2`.
// ===========================================================================
pub mod quant {
    use core::arch::x86_64::*;

    use crate::quant::Divisors;

    /// Quantize 64 i16 coefficients in natural order. Bit-exact
    /// equivalent to `arch::scalar::quant::quantize_natural`.
    ///
    /// Falls back to scalar at runtime if AVX2 is unavailable. The
    /// `is_x86_feature_detected!` check is cached after the first call.
    pub fn quantize_natural(block: &[i16; 64], div: &Divisors, out: &mut [i16; 64]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { quantize_avx2(block, div, out) }
        } else {
            crate::arch::scalar::quant::quantize_natural(block, div, out)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn quantize_avx2(block: &[i16; 64], div: &Divisors, out: &mut [i16; 64]) {
        unsafe {
            let block_p = block.as_ptr() as *const __m256i;
            let recip_p = div.recip.as_ptr() as *const __m256i;
            let corr_p = div.corr.as_ptr() as *const __m256i;
            let scale_p = div.scale.as_ptr() as *const __m256i;
            let out_p = out.as_mut_ptr() as *mut __m256i;

            // 4 ymm batches × 16 i16 lanes = 64 lanes = one block.
            for i in 0..4 {
                let x = _mm256_loadu_si256(block_p.add(i));
                let abs = _mm256_abs_epi16(x);
                let biased = _mm256_add_epi16(abs, _mm256_loadu_si256(corr_p.add(i)));
                // Stage 1: high-half multiply by reciprocal (≡ >> 16).
                let stage1 = _mm256_mulhi_epu16(biased, _mm256_loadu_si256(recip_p.add(i)));
                // Stage 2: high-half multiply by scale (≡ >> shift_remainder).
                let stage2 = _mm256_mulhi_epu16(stage1, _mm256_loadu_si256(scale_p.add(i)));
                // Restore sign of the original input — vpsignw negates
                // lanes where the second operand is negative, zeros where
                // it's zero, leaves alone where positive.
                let signed = _mm256_sign_epi16(stage2, x);
                _mm256_storeu_si256(out_p.add(i), signed);
            }
            // Avoid the AVX→SSE transition penalty for any downstream
            // SSE code that may run before the next AVX kernel.
            _mm256_zeroupper();
        }
    }
}

// ===========================================================================
// dct: AVX2 forward 8x8 integer LL&M DCT, in-place.
// Translated from `simd/x86_64/jfdctint-avx2.asm::jsimd_fdct_islow_avx2`.
// ===========================================================================
pub mod dct {
    use core::arch::x86_64::*;

    // 32-byte aligned constant blocks — same shape and ordering as the
    // upstream SEG_CONST tables.
    #[repr(C, align(32))]
    struct Aligned16<T>(T);

    // PW_F130_F054_MF130_F054:
    //   times 4 dw (F_0_541 + F_0_765),  F_0_541
    //   times 4 dw (F_0_541 - F_1_847),  F_0_541
    static PW_F130_F054_MF130_F054: Aligned16<[i16; 16]> = Aligned16([
        10703, 4433, 10703, 4433, 10703, 4433, 10703, 4433,
        -10704, 4433, -10704, 4433, -10704, 4433, -10704, 4433,
    ]);

    // PW_MF078_F117_F078_F117:
    //   times 4 dw (F_1_175 - F_1_961),  F_1_175
    //   times 4 dw (F_1_175 - F_0_390),  F_1_175
    static PW_MF078_F117_F078_F117: Aligned16<[i16; 16]> = Aligned16([
        -6436, 9633, -6436, 9633, -6436, 9633, -6436, 9633,
        6437, 9633, 6437, 9633, 6437, 9633, 6437, 9633,
    ]);

    // PW_MF060_MF089_MF050_MF256:
    //   times 4 dw (F_0_298 - F_0_899), -F_0_899
    //   times 4 dw (F_2_053 - F_2_562), -F_2_562
    static PW_MF060_MF089_MF050_MF256: Aligned16<[i16; 16]> = Aligned16([
        -4927, -7373, -4927, -7373, -4927, -7373, -4927, -7373,
        -4176, -20995, -4176, -20995, -4176, -20995, -4176, -20995,
    ]);

    // PW_F050_MF256_F060_MF089:
    //   times 4 dw (F_3_072 - F_2_562), -F_2_562
    //   times 4 dw (F_1_501 - F_0_899), -F_0_899
    static PW_F050_MF256_F060_MF089: Aligned16<[i16; 16]> = Aligned16([
        4177, -20995, 4177, -20995, 4177, -20995, 4177, -20995,
        4926, -7373, 4926, -7373, 4926, -7373, 4926, -7373,
    ]);

    // PD_DESCALE_P1 = 8 × (1 << (DESCALE_P1 - 1)) = 8 × 1024
    static PD_DESCALE_P1: Aligned16<[i32; 8]> = Aligned16([1024; 8]);

    // PD_DESCALE_P2 = 8 × (1 << (DESCALE_P2 - 1)) = 8 × 16384
    static PD_DESCALE_P2: Aligned16<[i32; 8]> = Aligned16([16384; 8]);

    // PW_DESCALE_P2X = 16 × (1 << (PASS1_BITS - 1)) = 16 × 2
    static PW_DESCALE_P2X: Aligned16<[i16; 16]> = Aligned16([2; 16]);

    // PW_1_NEG1: 8×1 then 8×-1 — used as the second operand of vpsignw
    // to flip the sign of the high 128 bits while leaving the low 128
    // alone (encoding "tmp10_neg11" packed swap).
    static PW_1_NEG1: Aligned16<[i16; 16]> = Aligned16([
        1, 1, 1, 1, 1, 1, 1, 1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    ]);

    /// 8x8 forward integer DCT (LL&M "islow"), in-place. Bit-exact
    /// equivalent to `arch::scalar::dct::fdct_islow`.
    pub fn fdct_islow(data: &mut [i16; 64]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { fdct_avx2(data) }
        } else {
            crate::arch::scalar::dct::fdct_islow(data)
        }
    }

    #[inline(always)]
    unsafe fn load(p: *const i16) -> __m256i {
        _mm256_loadu_si256(p as *const __m256i)
    }

    /// In-place 8x8x16-bit transpose. Mirrors the DOTRANSPOSE asm macro:
    /// 4 input ymm each holding two rows packed (low/high 128 bits) →
    /// 4 output ymm where each holds two columns packed.
    ///
    /// Caller must have AVX2 enabled (we rely on inlining into a
    /// `#[target_feature(enable = "avx2")]` function to satisfy the
    /// instruction-availability requirement).
    #[inline(always)]
    unsafe fn dotranspose(
        m1: __m256i,
        m2: __m256i,
        m3: __m256i,
        m4: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            // phase 1 — interleave 16-bit lanes
            let t5 = _mm256_unpacklo_epi16(m1, m2);
            let t6 = _mm256_unpackhi_epi16(m1, m2);
            let t7 = _mm256_unpacklo_epi16(m3, m4);
            let t8 = _mm256_unpackhi_epi16(m3, m4);

            // phase 2 — interleave 32-bit lanes
            let m1 = _mm256_unpacklo_epi32(t5, t7);
            let m2 = _mm256_unpackhi_epi32(t5, t7);
            let m3 = _mm256_unpacklo_epi32(t6, t8);
            let m4 = _mm256_unpackhi_epi32(t6, t8);

            // phase 3 — swap 64-bit halves to put columns in the right
            // 128-bit lanes
            let m1 = _mm256_permute4x64_epi64::<0x8D>(m1);
            let m2 = _mm256_permute4x64_epi64::<0x8D>(m2);
            let m3 = _mm256_permute4x64_epi64::<0xD8>(m3);
            let m4 = _mm256_permute4x64_epi64::<0xD8>(m4);

            (m1, m2, m3, m4)
        }
    }

    /// One 1-D DCT pass over 8 vectors-of-2-columns. Const generic on
    /// pass id (1 or 2) selects the descaling shift (DESCALE_P1=11 vs
    /// DESCALE_P2=15) and the small "PASS1_BITS round + shift" that is
    /// only present in pass 2.
    ///
    /// Returns `(data0_4, data3_1, data2_6, data7_5)`. Caller must
    /// have AVX2 enabled (relies on inlining; see `dotranspose`).
    #[inline(always)]
    unsafe fn dodct<const PASS: i32>(
        m1: __m256i,
        m2: __m256i,
        m3: __m256i,
        m4: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            // tmp values
            let m5 = _mm256_sub_epi16(m1, m4); // tmp6_7
            let m6 = _mm256_add_epi16(m1, m4); // tmp1_0
            let m7 = _mm256_add_epi16(m2, m3); // tmp3_2
            let m8 = _mm256_sub_epi16(m2, m3); // tmp4_5

            // -- Even part
            let m6 = _mm256_permute2x128_si256::<0x01>(m6, m6); // tmp0_1
            let m1 = _mm256_add_epi16(m6, m7); // tmp10_11
            let m6 = _mm256_sub_epi16(m6, m7); // tmp13_12

            let m7 = _mm256_permute2x128_si256::<0x01>(m1, m1); // tmp11_10
            let pw_1_neg1 = load(PW_1_NEG1.0.as_ptr());
            let m1 = _mm256_sign_epi16(m1, pw_1_neg1); // tmp10_neg11
            let m7 = _mm256_add_epi16(m7, m1); // (tmp10+tmp11)_(tmp10-tmp11)

            let m1 = if PASS == 1 {
                _mm256_slli_epi16::<2>(m7) // data0_4 (PASS1_BITS up-shift)
            } else {
                let pw_descale_p2x = load(PW_DESCALE_P2X.0.as_ptr());
                let m7 = _mm256_add_epi16(m7, pw_descale_p2x);
                _mm256_srai_epi16::<2>(m7) // data0_4 (PASS1_BITS down-shift)
            };

            // -- data2_6 (even part continued)
            let m7 = _mm256_permute2x128_si256::<0x01>(m6, m6); // tmp12_13
            let m2_lo = _mm256_unpacklo_epi16(m6, m7);
            let m6_hi = _mm256_unpackhi_epi16(m6, m7);

            let pw_f130 = load(PW_F130_F054_MF130_F054.0.as_ptr());
            let m2_lo = _mm256_madd_epi16(m2_lo, pw_f130);
            let m6_hi = _mm256_madd_epi16(m6_hi, pw_f130);

            let pd_descale = if PASS == 1 {
                _mm256_loadu_si256(PD_DESCALE_P1.0.as_ptr() as *const __m256i)
            } else {
                _mm256_loadu_si256(PD_DESCALE_P2.0.as_ptr() as *const __m256i)
            };
            let m2_lo = _mm256_add_epi32(m2_lo, pd_descale);
            let m6_hi = _mm256_add_epi32(m6_hi, pd_descale);
            let (m2_lo, m6_hi) = if PASS == 1 {
                (
                    _mm256_srai_epi32::<11>(m2_lo),
                    _mm256_srai_epi32::<11>(m6_hi),
                )
            } else {
                (
                    _mm256_srai_epi32::<15>(m2_lo),
                    _mm256_srai_epi32::<15>(m6_hi),
                )
            };

            let m3 = _mm256_packs_epi32(m2_lo, m6_hi); // data2_6

            // -- Odd part
            let m7 = _mm256_add_epi16(m8, m5); // z3_4

            let m2 = _mm256_permute2x128_si256::<0x01>(m7, m7); // z4_3
            let m6_lo = _mm256_unpacklo_epi16(m7, m2);
            let m7_hi = _mm256_unpackhi_epi16(m7, m2);

            let pw_mf078 = load(PW_MF078_F117_F078_F117.0.as_ptr());
            let m6_lo = _mm256_madd_epi16(m6_lo, pw_mf078); // z3_4L
            let m7_hi = _mm256_madd_epi16(m7_hi, pw_mf078); // z3_4H

            // -- data7_5
            let m4 = _mm256_permute2x128_si256::<0x01>(m5, m5); // tmp7_6
            let m2_lo = _mm256_unpacklo_epi16(m8, m4);
            let m4_hi = _mm256_unpackhi_epi16(m8, m4);

            let pw_mf060 = load(PW_MF060_MF089_MF050_MF256.0.as_ptr());
            let m2_lo = _mm256_madd_epi16(m2_lo, pw_mf060); // tmp4_5L
            let m4_hi = _mm256_madd_epi16(m4_hi, pw_mf060); // tmp4_5H

            let m2_lo = _mm256_add_epi32(m2_lo, m6_lo); // data7_5L
            let m4_hi = _mm256_add_epi32(m4_hi, m7_hi); // data7_5H

            let m2_lo = _mm256_add_epi32(m2_lo, pd_descale);
            let m4_hi = _mm256_add_epi32(m4_hi, pd_descale);
            let (m2_lo, m4_hi) = if PASS == 1 {
                (
                    _mm256_srai_epi32::<11>(m2_lo),
                    _mm256_srai_epi32::<11>(m4_hi),
                )
            } else {
                (
                    _mm256_srai_epi32::<15>(m2_lo),
                    _mm256_srai_epi32::<15>(m4_hi),
                )
            };

            let m4 = _mm256_packs_epi32(m2_lo, m4_hi); // data7_5

            // -- data3_1
            let m2 = _mm256_permute2x128_si256::<0x01>(m8, m8); // tmp5_4
            let m8_lo = _mm256_unpacklo_epi16(m5, m2);
            let m5_hi = _mm256_unpackhi_epi16(m5, m2);

            let pw_f050 = load(PW_F050_MF256_F060_MF089.0.as_ptr());
            let m8_lo = _mm256_madd_epi16(m8_lo, pw_f050); // tmp6_7L
            let m5_hi = _mm256_madd_epi16(m5_hi, pw_f050); // tmp6_7H

            let m8_lo = _mm256_add_epi32(m8_lo, m6_lo); // data3_1L
            let m5_hi = _mm256_add_epi32(m5_hi, m7_hi); // data3_1H

            let m8_lo = _mm256_add_epi32(m8_lo, pd_descale);
            let m5_hi = _mm256_add_epi32(m5_hi, pd_descale);
            let (m8_lo, m5_hi) = if PASS == 1 {
                (
                    _mm256_srai_epi32::<11>(m8_lo),
                    _mm256_srai_epi32::<11>(m5_hi),
                )
            } else {
                (
                    _mm256_srai_epi32::<15>(m8_lo),
                    _mm256_srai_epi32::<15>(m5_hi),
                )
            };

            let m2 = _mm256_packs_epi32(m8_lo, m5_hi); // data3_1

            (m1, m2, m3, m4)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn fdct_avx2(data: &mut [i16; 64]) {
        unsafe {
            let p = data.as_mut_ptr() as *mut __m256i;
            // Load 4 ymm: each carries 2 rows of 8 i16.
            let m4 = _mm256_loadu_si256(p);
            let m5 = _mm256_loadu_si256(p.add(1));
            let m6 = _mm256_loadu_si256(p.add(2));
            let m7 = _mm256_loadu_si256(p.add(3));

            // Re-pack so each ymm holds rows N and N+4 (lo/hi 128).
            let m0 = _mm256_permute2x128_si256::<0x20>(m4, m6);
            let m1 = _mm256_permute2x128_si256::<0x31>(m4, m6);
            let m2 = _mm256_permute2x128_si256::<0x20>(m5, m7);
            let m3 = _mm256_permute2x128_si256::<0x31>(m5, m7);

            // Pass 1: rows.
            let (t0, t1, t2, t3) = dotranspose(m0, m1, m2, m3);
            let (out0, out1, out2, out3) = dodct::<1>(t0, t1, t2, t3);
            // out0 = data0_4, out1 = data3_1, out2 = data2_6, out3 = data7_5

            // Re-pack between passes: collect the diagonal pairs.
            let p4 = _mm256_permute2x128_si256::<0x20>(out1, out3); // data3_7
            let p1 = _mm256_permute2x128_si256::<0x31>(out1, out3); // data1_5

            // Pass 2: columns.
            let (t0, t1, t2, t3) = dotranspose(out0, p1, out2, p4);
            let (out0, out1, out2, out3) = dodct::<2>(t0, t1, t2, t3);

            // Final repack into row order and store.
            let s0 = _mm256_permute2x128_si256::<0x30>(out0, out1); // data0_1
            let s1 = _mm256_permute2x128_si256::<0x20>(out2, out1); // data2_3
            let s2 = _mm256_permute2x128_si256::<0x31>(out0, out3); // data4_5
            let s3 = _mm256_permute2x128_si256::<0x21>(out2, out3); // data6_7

            _mm256_storeu_si256(p, s0);
            _mm256_storeu_si256(p.add(1), s1);
            _mm256_storeu_si256(p.add(2), s2);
            _mm256_storeu_si256(p.add(3), s3);

            _mm256_zeroupper();
        }
    }
}

// ===========================================================================
// Cross-check tests — only run on x86_64 builds, where both the AVX2
// quantize and the scalar reference are reachable.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::scalar;

    #[test]
    fn quant_avx2_matches_scalar_random() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            // No AVX2 on this CPU — runtime dispatch will use scalar
            // anyway, so there is nothing to cross-check.
            return;
        }

        use crate::quant::build_divisors;
        use crate::tables::{STD_LUMA_QUANT, scale_quant_table};

        let mut block = [0i16; 64];
        for (i, v) in block.iter_mut().enumerate() {
            let m = (i as i32 * 37) % 4001 - 2000;
            *v = m as i16;
        }
        let qtab = scale_quant_table(&STD_LUMA_QUANT, 80);
        let div = build_divisors(&qtab);

        let mut sout = [0i16; 64];
        let mut aout = [0i16; 64];
        scalar::quant::quantize_natural(&block, &div, &mut sout);
        quant::quantize_natural(&block, &div, &mut aout);
        assert_eq!(sout, aout);
    }

    fn random_block(seed: u64) -> [i16; 64] {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut b = [0i16; 64];
        for v in &mut b {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) as i32 % 256 - 128) as i16;
        }
        b
    }

    #[test]
    fn fdct_avx2_matches_scalar_zeros() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut a = [0i16; 64];
        let mut b = [0i16; 64];
        scalar::dct::fdct_islow(&mut a);
        dct::fdct_islow(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn fdct_avx2_matches_scalar_const() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut a = [42i16; 64];
        let mut b = [42i16; 64];
        scalar::dct::fdct_islow(&mut a);
        dct::fdct_islow(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn fdct_avx2_matches_scalar_ramp() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
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
    fn fdct_avx2_matches_scalar_random() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for seed in 0..5u64 {
            let mut a = random_block(seed);
            let mut b = a;
            scalar::dct::fdct_islow(&mut a);
            dct::fdct_islow(&mut b);
            assert_eq!(a, b, "seed={seed}");
        }
    }

    #[test]
    fn fdct_avx2_matches_scalar_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
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
    fn quant_avx2_matches_scalar_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        use crate::quant::build_divisors;

        // All-zero block: quantize → all zero on both backends.
        let block_zero = [0i16; 64];
        // Block with i16 extremes alternating sign.
        let mut block_extreme = [0i16; 64];
        for (i, v) in block_extreme.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 32767 } else { -32768 };
        }

        // Try a few quality levels (different divisor magnitudes).
        for q in [10u8, 50, 80, 95] {
            let qtab = crate::tables::scale_quant_table(&crate::tables::STD_LUMA_QUANT, q);
            let div = build_divisors(&qtab);
            for block in [&block_zero, &block_extreme] {
                let mut s = [0i16; 64];
                let mut a = [0i16; 64];
                scalar::quant::quantize_natural(block, &div, &mut s);
                quant::quantize_natural(block, &div, &mut a);
                assert_eq!(s, a, "q={q}");
            }
        }
    }
}
