//! Quantization (libjpeg-turbo reciprocal-multiply scheme) and zig-zag
//! reordering.
//!
//! With the integer LL&M DCT, post-DCT coefficients are scaled by 8
//! relative to the true DCT. We absorb that factor by quantizing
//! against `quant_table[i] << 3` rather than `quant_table[i]`. The
//! division itself is done by precomputing a reciprocal-multiply tuple
//! per coefficient (recip / corr / shift), exactly as libjpeg-turbo does
//! in `jcdctmgr.c::compute_reciprocal`. This makes the scalar and NEON
//! quantization paths bit-exact.
//!
//! Layout of `Divisors` (4 × 64 × i16 = 512 bytes):
//!     [0..64)    reciprocals (interpreted as u16 at use)
//!     [64..128)  correction biases (u16)
//!     [128..192) "scale" — unused in our scalar path but reserved for
//!                NEON's `vqdmulhq` variant (we use the wider mul→shrn
//!                pattern instead, mirroring libjpeg's WITH_SIMD path)
//!     [192..256) shift amount (i16, signed because some entries are 0)

// On aarch64 the scalar `quantize_and_zigzag` is used only by tests;
// suppress the otherwise-correct dead-code warnings.
#![allow(dead_code)]

use crate::tables::ZIGZAG;

/// Per-coefficient quantization helper. Indexed in *natural* order
/// (NOT zig-zag). The `quantize_and_zigzag` function reorders on output.
#[derive(Clone, Copy)]
pub struct Divisors {
    pub recip: [u16; 64],
    pub corr: [u16; 64],
    pub shift: [i16; 64],
}

/// Build the divisor table for one component. `quant` is the standard
/// 8-bit DQT values (already scaled by quality). The `<<3` here is what
/// folds the LL&M DCT's overall scale of 8 into the divide.
pub fn build_divisors(quant: &[u8; 64]) -> Divisors {
    let mut d = Divisors { recip: [0; 64], corr: [0; 64], shift: [0; 64] };
    for (i, &q) in quant.iter().enumerate() {
        let divisor = (q as u32) << 3;
        let (recip, corr, shift) = compute_reciprocal(divisor);
        d.recip[i] = recip;
        d.corr[i] = corr;
        d.shift[i] = shift;
    }
    d
}

/// Compute a 16-bit reciprocal/correction/shift triple such that
/// `((x + corr) * recip) >> (shift + 16)` equals `(x + divisor/2) /
/// divisor` for all `x` in `0..=2^15`.
///
/// This is a port of libjpeg-turbo's `compute_reciprocal` specialized
/// for `sizeof(DCTELEM) == 2` (16-bit). See `jcdctmgr.c` for the
/// derivation. We always go through the SIMD-shaped path (recip < 2^16,
/// shift relative to a 16-bit narrow) — our scalar quantize uses the
/// same pre-shift pattern so divisors are interchangeable.
fn compute_reciprocal(divisor: u32) -> (u16, u16, i16) {
    if divisor <= 1 {
        // Identity quantization: dividing by 1 is the same as no op.
        // We encode that as recip=1, corr=0, shift=-16 so that
        // (x + 0) * 1 >> (16 + -16) = x.
        return (1, 0, -16);
    }
    // b = position of MSB of `divisor` (0-based, so divisor in [1,2] →
    // b=0, [2,4) → b=1, etc.). flss-1 in libjpeg.
    let b = 31 - divisor.leading_zeros();
    // r = sizeof(DCTELEM)*8 + b = 16 + b.
    let r = 16u32 + b;

    let mut fq = (1u64 << r) / divisor as u64;
    let fr = (1u64 << r) % divisor as u64;
    let mut c = (divisor / 2) as u64;
    let mut r_eff = r;
    if fr == 0 {
        // Power-of-two divisor: fq is one bit too big, drop one.
        fq >>= 1;
        r_eff -= 1;
    } else if fr <= (divisor as u64 / 2) {
        c += 1;
    } else {
        fq += 1;
    }

    let recip = fq as u16; // fits because fq < 2^16 by construction.
    let corr = c as u16;
    let shift = r_eff as i32 - 16;
    (recip, corr, shift as i16)
}

/// Scalar quantize using `Divisors`, reordered to zig-zag scan order.
///
/// Equivalent to libjpeg-turbo's `quantize()` in `jcdctmgr.c`:
///
///   abs_temp = abs(temp); product = (abs_temp + corr) * recip;
///   temp = product >> (shift + 16); apply sign.
///
/// This is what the NEON path emulates, so the two are bit-exact.
pub fn quantize_and_zigzag(
    block: &[i16; 64],
    div: &Divisors,
) -> [i16; 64] {
    // Pass 1: compute the natural-order quantized block (matches what
    // libjpeg/NEON would write to `coef_block`).
    let mut natural = [0i16; 64];
    for (i, &b) in block.iter().enumerate() {
        natural[i] = quantize_one(b, div.recip[i], div.corr[i], div.shift[i]);
    }
    // Pass 2: zig-zag reorder for entropy coding.
    let mut zz = [0i16; 64];
    for (k, &nat) in ZIGZAG.iter().enumerate() {
        zz[k] = natural[nat];
    }
    zz
}

#[inline(always)]
fn quantize_one(temp: i16, recip: u16, corr: u16, shift: i16) -> i16 {
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

// ===========================================================================
// AArch64 NEON quantize. Translated from libjpeg-turbo's
// `simd/arm/jquanti-neon.c::jsimd_quantize_neon`. Operates on a natural-
// order block in place; we apply a separate scalar zig-zag reorder
// afterward (zig-zag itself is not vectorizable usefully).
// ===========================================================================
#[cfg(target_arch = "aarch64")]
pub fn quantize_and_zigzag_neon(block: &[i16; 64], div: &Divisors) -> [i16; 64] {
    let mut natural = [0i16; 64];
    unsafe { neon::quantize_neon(block, div, &mut natural) };
    let mut zz = [0i16; 64];
    for (k, &nat) in ZIGZAG.iter().enumerate() {
        zz[k] = natural[nat];
    }
    zz
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Divisors;
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn quantize_neon(
        workspace: &[i16; 64],
        div: &Divisors,
        out: &mut [i16; 64],
    ) { unsafe {
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
    }}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_round_trip_small() {
        // For typical small divisors, (x + d/2) / d should match the
        // reciprocal-multiply form for all x in our coefficient range.
        for d in [1u32, 8, 11 * 8, 99 * 8, 255 * 8] {
            let (recip, corr, shift) = compute_reciprocal(d);
            for x in -8192..=8192i32 {
                let want = if d <= 1 {
                    x
                } else {
                    let abs = x.unsigned_abs();
                    let q = ((abs + d / 2) / d) as i32;
                    if x < 0 { -q } else { q }
                };
                let got = quantize_one(x as i16, recip, corr, shift) as i32;
                // Tiny accuracy-vs-reciprocal disagreement is allowed
                // only at the rounding boundary; for the range we care
                // about the shift is exact.
                if want.abs() <= i16::MAX as i32 {
                    assert_eq!(got, want, "d={d} x={x}");
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_quantize_matches_scalar() {
        // Random-ish coefficients across a few quantization tables.
        let mut block = [0i16; 64];
        for (i, v) in block.iter_mut().enumerate() {
            // Vary sign, magnitude.
            let m = (i as i32 * 37) % 4001 - 2000;
            *v = m as i16;
        }
        let qtab = crate::tables::scale_quant_table(&crate::tables::STD_LUMA_QUANT, 80);
        let div = build_divisors(&qtab);

        let scalar = quantize_and_zigzag(&block, &div);
        let neon = quantize_and_zigzag_neon(&block, &div);
        assert_eq!(scalar, neon);
    }
}
