//! JPEG standard tables (Annex K).
//!
//! These are the reference quantization and Huffman tables from
//! ITU-T T.81 Annex K. Every conventional baseline encoder ships
//! these verbatim; a decoder that reads our DQT/DHT segments will
//! find the same numeric values it would in libjpeg's output.
//!
//! The tables are pure data; the SIMD kernels operate on the
//! *application* of these tables (DCT, quantize, Huffman bit-pack),
//! not on the table contents.

// ---------------------------------------------------------------
// Zig-zag scan order (Annex F.1.1.5).
//
// 8x8 natural-order indices traversed in the diagonal "zig-zag" pattern
// the JPEG spec uses to put low-frequency DCT coefficients first
// (where, after quantization, most of the non-zero values live; AC
// run-length coding compresses the trailing zeros efficiently).
// ---------------------------------------------------------------
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// ---------------------------------------------------------------
// Annex K.1 / K.2 — sample quantization tables.
//
// These are the values that ship in nearly every JPEG ever produced
// at the "default" quality of 50. The runtime quality knob scales
// these values (see `scale_quant_table`).
//
// Layout: natural row-major (NOT zig-zag). We zig-zag at write time.
// ---------------------------------------------------------------

pub const STD_LUMA_QUANT: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

pub const STD_CHROMA_QUANT: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// Map a 1..=100 quality knob to the libjpeg "scale factor" the
/// reference encoder uses. Lifted directly from libjpeg's
/// `jpeg_quality_scaling`. q=50 returns 100 (no scaling); q=100
/// returns 0 which we treat as "all entries become 1".
fn quality_to_scale(quality: u8) -> u32 {
    let q = quality.clamp(1, 100) as u32;
    if q < 50 { 5000 / q } else { 200 - q * 2 }
}

/// Apply the scale factor to a base quant table and clamp into
/// `[1, 255]` (baseline 8-bit DQT). Same arithmetic libjpeg uses in
/// `jpeg_add_quant_table`.
pub fn scale_quant_table(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let scale = quality_to_scale(quality);
    let mut out = [0u8; 64];
    for (i, &v) in base.iter().enumerate() {
        let scaled = ((v as u32 * scale) + 50) / 100;
        out[i] = scaled.clamp(1, 255) as u8;
    }
    out
}

// ---------------------------------------------------------------
// Annex K.3 / K.4 / K.5 / K.6 — standard Huffman tables.
//
// Stored in the form a DHT segment carries:
//   `bits[i]` = number of codes of length `i+1` (16 entries)
//   `values[]`  = symbols, in canonical order
// At runtime we expand these into a (code, length) lookup table
// (see `huffman::HuffmanTable`).
// ---------------------------------------------------------------

pub struct StdHuffman {
    pub bits: [u8; 16],
    pub values: &'static [u8],
}

// K.3: Luma DC.
pub const STD_LUMA_DC: StdHuffman = StdHuffman {
    bits: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

// K.5: Chroma DC.
pub const STD_CHROMA_DC: StdHuffman = StdHuffman {
    bits: [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    values: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

// K.4: Luma AC.
pub const STD_LUMA_AC: StdHuffman = StdHuffman {
    bits: [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
    values: &[
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
        0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52,
        0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
        0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
        0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
        0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
        0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3,
        0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
        0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ],
};

// K.6: Chroma AC.
pub const STD_CHROMA_AC: StdHuffman = StdHuffman {
    bits: [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
    values: &[
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
        0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33,
        0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
        0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
        0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63,
        0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
        0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
        0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
        0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
        0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ],
};

// ---------------------------------------------------------------
// Quantization divisor machinery (libjpeg-turbo reciprocal-multiply).
//
// This is shared core data consumed by every kernel backend
// (`crate::arch::{scalar,neon,x86_64}`), so it lives down here in the
// shared `tables` layer rather than up in `encode`. The encode-side
// orchestration (`quantize_and_zigzag`) stays in `encode::quant` and
// imports `Divisors` from here.
//
// With the integer LL&M DCT, post-DCT coefficients are scaled by 8
// relative to the true DCT. We absorb that factor by quantizing
// against `quant_table[i] << 3` rather than `quant_table[i]`. The
// division itself is done by precomputing a reciprocal-multiply tuple
// per coefficient (recip / corr / shift), exactly as libjpeg-turbo
// does in `jcdctmgr.c::compute_reciprocal`.
//
// Layout of `Divisors`:
//     recip[64]  reciprocals (u16)
//     corr[64]   correction biases (u16, ≈ d/2 with rounding adjustment)
//     scale[64]  AVX2-only secondary multiplier (u16, = `1 << (32 - r)`)
//     shift[64]  scalar/NEON post-shift (i16, signed for the d=1 identity)
//
// Backends use different subsets:
//   - scalar / NEON: `(|x|+corr) * recip >> 16, then >> shift`
//   - AVX2:          `(|x|+corr) * recip >> 16, then * scale >> 16`
//
// The two are algebraically equivalent because `scale = 1 << (16-shift)`
// (so `* scale >> 16` and `>> shift` produce the same result modulo
// the high 16 bits being implicitly chopped by the multiply, which is
// exactly the `vpmulhuw` semantics).
// ---------------------------------------------------------------

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
