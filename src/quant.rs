//! Quantization (libjpeg-turbo reciprocal-multiply scheme) and zig-zag
//! reordering.
//!
//! The hot quantize kernel lives in `crate::arch::backend::quant` so
//! that scalar and SIMD backends share a single bit-exact contract.
//! Zig-zag itself is just a permutation and is applied here in scalar
//! after the kernel writes a natural-order block.
//!
//! With the integer LL&M DCT, post-DCT coefficients are scaled by 8
//! relative to the true DCT. We absorb that factor by quantizing
//! against `quant_table[i] << 3` rather than `quant_table[i]`. The
//! division itself is done by precomputing a reciprocal-multiply tuple
//! per coefficient (recip / corr / shift), exactly as libjpeg-turbo
//! does in `jcdctmgr.c::compute_reciprocal`.
//!
//! Layout of `Divisors`:
//!     recip[64]  reciprocals (u16)
//!     corr[64]   correction biases (u16, ≈ d/2 with rounding adjustment)
//!     scale[64]  AVX2-only secondary multiplier (u16, = `1 << (32 - r)`)
//!     shift[64]  scalar/NEON post-shift (i16, signed for the d=1 identity)
//!
//! Backends use different subsets:
//!   - scalar / NEON: `(|x|+corr) * recip >> 16, then >> shift`
//!   - AVX2:          `(|x|+corr) * recip >> 16, then * scale >> 16`
//!
//! The two are algebraically equivalent because `scale = 1 << (16-shift)`
//! (so `* scale >> 16` and `>> shift` produce the same result modulo
//! the high 16 bits being implicitly chopped by the multiply, which is
//! exactly the `vpmulhuw` semantics).

use crate::arch;
use crate::tables::ZIGZAG;

/// Per-coefficient quantization helper. Indexed in *natural* order
/// (NOT zig-zag). The `quantize_and_zigzag` function reorders on output.
///
/// `Debug` is derived for diagnostics, but the default `[u16; 64] × 3 +
/// [i16; 64]` formatting is verbose; expect ~256 lines of integers if
/// you actually print one.
#[derive(Clone, Copy, Debug)]
pub struct Divisors {
    pub recip: [u16; 64],
    pub corr: [u16; 64],
    pub scale: [u16; 64],
    pub shift: [i16; 64],
}

/// Build the divisor table for one component. `quant` is the standard
/// 8-bit DQT values (already scaled by quality). The `<<3` here is
/// what folds the LL&M DCT's overall scale of 8 into the divide.
pub fn build_divisors(quant: &[u8; 64]) -> Divisors {
    let mut d = Divisors {
        recip: [0; 64],
        corr: [0; 64],
        scale: [0; 64],
        shift: [0; 64],
    };
    for (i, &q) in quant.iter().enumerate() {
        let divisor = (q as u32) << 3;
        let (recip, corr, scale, shift) = compute_reciprocal(divisor);
        d.recip[i] = recip;
        d.corr[i] = corr;
        d.scale[i] = scale;
        d.shift[i] = shift;
    }
    d
}

/// Compute the (recip, corr, scale, shift) tuple such that
///   `((|x| + corr) * recip) >> 16) >> shift`         (scalar / NEON)
/// and
///   `(((|x| + corr) * recip) >> 16) * scale) >> 16`  (AVX2 via vpmulhuw)
/// both equal `(|x| + divisor/2) / divisor` for every `|x|` in `0..=2^15`.
///
/// Port of libjpeg-turbo's `compute_reciprocal` for `sizeof(DCTELEM)
/// == 2`. The two backends differ only in how they apply the residual
/// post-`>>16` shift: scalar/NEON have hardware variable-shift, AVX2
/// fakes it with another high-half multiply (`scale = 1 << (32 - r)`).
fn compute_reciprocal(divisor: u32) -> (u16, u16, u16, i16) {
    if divisor <= 1 {
        // Identity quantization: dividing by 1 is the same as no op.
        // Scalar / NEON encode as recip=1, corr=0, shift=-16.
        // AVX2 path uses scale=1 (irrelevant; the libjpeg-turbo C code
        // marks this case as "scale is irrelevant" because the C path
        // is selected for d=1 even in WITH_SIMD builds).
        return (1, 0, 1, -16);
    }
    // b = position of MSB of `divisor` (0-based, so divisor in [1,2] →
    // b=0, [2,4) → b=1, etc.). `flss(d) - 1` in libjpeg.
    let b = 31 - divisor.leading_zeros();
    // r = sizeof(DCTELEM)*8 + b = 16 + b.
    let r0 = 16u32 + b;

    let mut fq = (1u64 << r0) / divisor as u64;
    let fr = (1u64 << r0) % divisor as u64;
    let mut c = (divisor / 2) as u64;
    let mut r = r0;
    if fr == 0 {
        // Power-of-two divisor: fq is one bit too big, drop one.
        fq >>= 1;
        r -= 1;
    } else if fr <= (divisor as u64 / 2) {
        c += 1;
    } else {
        fq += 1;
    }

    let recip = fq as u16; // fits because fq < 2^16 by construction.
    let corr = c as u16;
    // scale = 1 << (sizeof(DCTELEM)*8*2 - r) = 1 << (32 - r). Since
    // `(q << 3) >= 8` in our caller, `b >= 3` (max b=10 for q=255),
    // so `r ∈ [18, 26]` and `scale ∈ [64, 16384]` — fits in u16.
    let scale = (1u32 << (32 - r)) as u16;
    let shift = r as i32 - 16;
    (recip, corr, scale, shift as i16)
}

/// Quantize a block (in natural order) and emit zig-zag-reordered output.
pub fn quantize_and_zigzag(block: &[i16; 64], div: &Divisors) -> [i16; 64] {
    let mut natural = [0i16; 64];
    arch::backend::quant::quantize_natural(block, div, &mut natural);
    let mut zz = [0i16; 64];
    for (k, &nat) in ZIGZAG.iter().enumerate() {
        zz[k] = natural[nat];
    }
    zz
}
