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
//! Layout of `Divisors` (4 × 64 × i16 = 512 bytes):
//!     [0..64)    reciprocals (interpreted as u16 at use)
//!     [64..128)  correction biases (u16)
//!     [128..192) "scale" — unused in our scalar path but reserved for
//!                NEON's `vqdmulhq` variant (we use the wider mul→shrn
//!                pattern instead, mirroring libjpeg's WITH_SIMD path)
//!     [192..256) shift amount (i16, signed because some entries are 0)

use crate::arch;
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
/// 8-bit DQT values (already scaled by quality). The `<<3` here is
/// what folds the LL&M DCT's overall scale of 8 into the divide.
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
/// Port of libjpeg-turbo's `compute_reciprocal` for `sizeof(DCTELEM)
/// == 2`. We always go through the SIMD-shaped path (recip < 2^16,
/// shift relative to a 16-bit narrow); our scalar quantize uses the
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
