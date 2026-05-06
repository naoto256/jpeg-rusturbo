//! x86_64 SIMD kernels — translations of libjpeg-turbo's
//! `simd/x86_64/*-avx2.asm`. See `LICENSES/libjpeg-turbo.txt`.
//!
//! Backend status (incremental port):
//!   - `quant`    — AVX2 implementation (this commit)
//!   - `color`    — delegate to scalar (TODO)
//!   - `dct`      — delegate to scalar (TODO)
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
pub mod dct {
    pub use crate::arch::scalar::dct::*;
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
