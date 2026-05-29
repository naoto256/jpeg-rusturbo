//! Quantization (libjpeg-turbo reciprocal-multiply scheme) and zig-zag
//! reordering.
//!
//! The hot quantize kernel lives in `crate::arch::backend::quant` so
//! that scalar and SIMD backends share a single bit-exact contract.
//! Zig-zag itself is just a permutation and is applied here in scalar
//! after the kernel writes a natural-order block.
//!
//! The shared divisor machinery (`Divisors` / `build_divisors`) lives
//! in `crate::tables` so that the kernel backends can depend on it
//! without reaching up into the `encode` layer. This module keeps the
//! encode-side orchestration (`quantize_and_zigzag`).

use crate::arch;
use crate::tables::Divisors;

/// Quantize a block (in natural order) and emit zig-zag-reordered output.
pub fn quantize_and_zigzag(block: &[i16; 64], div: &Divisors) -> [i16; 64] {
    let mut natural = [0i16; 64];
    arch::backend::quant::quantize_natural(block, div, &mut natural);
    let mut zz = [0i16; 64];
    arch::backend::quant::zigzag_scatter(&natural, &mut zz);
    zz
}
