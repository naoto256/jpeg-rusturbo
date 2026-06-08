//! x86_64 SIMD kernels — translations of libjpeg-turbo's
//! `simd/x86_64/*-avx2.asm`. See `NOTICE.md`.
//!
//! Backend status:
//!
//! - `quant` — AVX2
//! - `dct` — AVX2
//! - `color::rgb_row_to_ycc` — AVX2 for n=16 (4-byte RGBA-family and
//!   3-byte RGB/BGR); scalar fallback for n=8 / non-AVX2
//! - `color::h2v2_downsample` — AVX2
//! - `color::h2v1_downsample` — AVX2
//! - `huffman::nonzero_bitmap` — SSE2 (`pcmpeqw + packsswb + pmovmskb`,
//!   translated from `jchuff-sse2.asm`)
//!
//! Runtime feature detection: AVX2 is the only target we ever dispatch
//! to from x86_64. Non-AVX2 CPUs hit the scalar fallback at runtime via
//! `is_x86_feature_detected!`. The result is cached after first call.
//!
//! # Safety
//!
//! Every `unsafe { … }` block in this module wraps a `core::arch::x86_64::*`
//! intrinsic. The module is reached only when both compile-time
//! `#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]` and
//! the runtime guard hold:
//!
//! - **AVX2 paths** (`color`, `dct`, `quant`) are dispatched only when
//!   `is_x86_feature_detected!("avx2")` returns true. The check is performed
//!   once on first call from `arch::backend::*` and cached; non-AVX2 CPUs
//!   never reach these intrinsics. `__m256i` / AVX2 instructions are
//!   therefore valid at the CPU level.
//! - **SSE2 path** (`huffman::nonzero_bitmap`) is unconditional on
//!   `x86_64` because SSE2 is in the x86_64-v1 baseline and is always
//!   available.
//!
//! On the Rust side, each call takes pointers / slice references whose
//! lifetimes cover the load/store window and whose element counts match
//! the intrinsic's vector width (16-lane i16 for SSE2; 32-lane u8 /
//! 16-lane i16 / 8-lane i32 for AVX2). No call reads past the end of a
//! borrow or writes to an aliased destination — the cross-check tests at
//! the bottom of this file (and the equivalent in `tests/`) would have
//! flagged any such bug by comparing every emitted byte against the
//! scalar reference.

#![allow(dead_code)]

// ===========================================================================
// huffman: SSE2 nonzero bitmap for AC scan
// ===========================================================================
pub mod huffman {
    use core::arch::x86_64::*;

    /// Bit `k` is set iff `block[k] != 0`. Translated from
    /// libjpeg-turbo's `simd/x86_64/jchuff-sse2.asm` bitmap step:
    /// `pcmpeqw + packsswb + pmovmskb` over four 16-coefficient
    /// chunks, NOT the result for "nonzero" semantics, OR into a u64.
    ///
    /// SSE2 is part of the x86_64 baseline so no runtime feature gate
    /// is required.
    pub fn nonzero_bitmap(block: &[i16; 64]) -> u64 {
        unsafe { nonzero_bitmap_sse2(block) }
    }

    /// # Safety
    /// SSE2 is unconditionally available on `target_arch = "x86_64"`.
    /// `block` is a fixed-size reference; no caller-side invariants.
    #[target_feature(enable = "sse2")]
    unsafe fn nonzero_bitmap_sse2(block: &[i16; 64]) -> u64 {
        unsafe {
            let zero = _mm_setzero_si128();
            let mut bm: u64 = 0;
            for chunk in 0..4 {
                let p = block.as_ptr().add(chunk * 16) as *const __m128i;
                let v0 = _mm_loadu_si128(p);
                let v1 = _mm_loadu_si128(p.add(1));
                // 0xFFFF per i16 lane if zero, else 0x0000.
                let eq0 = _mm_cmpeq_epi16(v0, zero);
                let eq1 = _mm_cmpeq_epi16(v1, zero);
                // Signed-saturate pack i16→i8: 0xFFFF (-1) → 0xFF, 0 → 0.
                let packed = _mm_packs_epi16(eq0, eq1);
                // Extract MSB of each byte ⇒ "is zero" mask.
                let zero_mask = _mm_movemask_epi8(packed) as u32 & 0xFFFF;
                // Invert for "nonzero" semantics.
                let nz_mask = !zero_mask & 0xFFFF;
                bm |= (nz_mask as u64) << (chunk * 16);
            }
            bm
        }
    }

    /// Precompute the JPEG magnitude category (`size`) and the
    /// magnitude bits for every coefficient. Routed to the scalar form
    /// for now — an AVX2 SIMD version is feasible (PSRLW + sign mask +
    /// lzcnt-like pattern via PSHUFB lookup) but `encode_block` self-
    /// time isn't yet hot enough on Cascade Lake to justify it. Revisit
    /// when profiling shows otherwise.
    pub fn ac_magnitudes(block: &[i16; 64], sizes: &mut [u8; 64], bits_lut: &mut [u16; 64]) {
        crate::arch::scalar::huffman::ac_magnitudes(block, sizes, bits_lut)
    }
}

// ===========================================================================
// encode: x86_64 full-MCU encoder front-half hooks.
// ===========================================================================
pub mod encode {
    use core::arch::x86_64::*;

    use crate::color::RGB;
    use crate::tables::Divisors;

    /// Full-MCU RGB 4:2:0 front-half hook for x86_64.
    ///
    /// This keeps the higher-level encode pipeline shape identical to
    /// the scalar reference, but avoids materializing the 16x16 luma
    /// plane before splitting it into four 8x8 blocks. The hot kernels
    /// below (`rgb_row_to_ycc`, `h2v2_downsample`, `fdct_islow`,
    /// `quantize_natural`) still dispatch to AVX2 when available.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_mcu_420_rgb_full(
        pixels: &[u8],
        width: u32,
        x0: u32,
        y0: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        out: &mut [[i16; 64]],
    ) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                return quantize_mcu_420_rgb_full_avx2(
                    pixels, width, x0, y0, div_luma, div_chroma, out,
                );
            }
        }
        quantize_mcu_420_rgb_full_scalar_kernels(pixels, width, x0, y0, div_luma, div_chroma, out);
    }

    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_mcu_420_rgb_full_avx2(
        pixels: &[u8],
        width: u32,
        x0: u32,
        y0: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        out: &mut [[i16; 64]],
    ) {
        unsafe {
            quantize_mcu_420_rgb_full_avx2_fused(pixels, width, x0, y0, div_luma, div_chroma, out);
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn quantize_mcu_420_rgb_full_avx2_fused(
        pixels: &[u8],
        width: u32,
        x0: u32,
        y0: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        out: &mut [[i16; 64]],
    ) {
        unsafe {
            debug_assert!(x0 + 16 <= width);
            debug_assert!(out.len() >= 6);

            let stride = width as usize * RGB.bpp;
            let mut cb_blk = [0i16; 64];
            let mut cr_blk = [0i16; 64];

            for pair in 0..8usize {
                let j0 = pair * 2;
                let j1 = j0 + 1;

                let row0 = (y0 as usize + j0) * stride + x0 as usize * RGB.bpp;
                let row1 = (y0 as usize + j1) * stride + x0 as usize * RGB.bpp;

                let block_row0 = j0 & 7;
                let block_row1 = j1 & 7;
                let top0 = j0 < 8;
                let top1 = j1 < 8;

                let left0 = if top0 { 0 } else { 2 };
                let left1 = if top1 { 0 } else { 2 };

                let (cb0, cr0) = super::color::rgb24_16_avx2_to_luma_chroma_vectors(
                    pixels.as_ptr().add(row0),
                    RGB,
                    (*out.as_mut_ptr().add(left0))
                        .as_mut_ptr()
                        .add(block_row0 * 8),
                    (*out.as_mut_ptr().add(left0 + 1))
                        .as_mut_ptr()
                        .add(block_row0 * 8),
                );
                let (cb1, cr1) = super::color::rgb24_16_avx2_to_luma_chroma_vectors(
                    pixels.as_ptr().add(row1),
                    RGB,
                    (*out.as_mut_ptr().add(left1))
                        .as_mut_ptr()
                        .add(block_row1 * 8),
                    (*out.as_mut_ptr().add(left1 + 1))
                        .as_mut_ptr()
                        .add(block_row1 * 8),
                );

                if pair == 3 {
                    fdct_quantize_zigzag::<true>(&mut out[0], div_luma);
                    fdct_quantize_zigzag::<true>(&mut out[1], div_luma);
                }

                downsample_chroma_pair(cb0, cb1, cb_blk.as_mut_ptr().add(pair * 8));
                downsample_chroma_pair(cr0, cr1, cr_blk.as_mut_ptr().add(pair * 8));
            }

            fdct_quantize_zigzag::<true>(&mut out[2], div_luma);
            fdct_quantize_zigzag::<true>(&mut out[3], div_luma);
            fdct_quantize_zigzag_into::<true>(&mut cb_blk, div_chroma, &mut out[4]);
            fdct_quantize_zigzag_into::<true>(&mut cr_blk, div_chroma, &mut out[5]);
        }
    }

    #[inline(always)]
    unsafe fn downsample_chroma_pair(row0: __m256i, row1: __m256i, dst: *mut i16) {
        unsafe {
            let ones = _mm256_set1_epi16(1);
            let bias = _mm256_setr_epi32(1, 2, 1, 2, 1, 2, 1, 2);
            let level_shift = _mm256_set1_epi32(128);

            let pairs0 = _mm256_madd_epi16(row0, ones);
            let pairs1 = _mm256_madd_epi16(row1, ones);
            let avg =
                _mm256_srli_epi32::<2>(_mm256_add_epi32(_mm256_add_epi32(pairs0, pairs1), bias));
            let signed = _mm256_sub_epi32(avg, level_shift);

            let packed = _mm256_packs_epi32(signed, signed);
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256::<1>(packed);
            let row = _mm_unpacklo_epi64(lo, hi);
            _mm_storeu_si128(dst as *mut __m128i, row);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn quantize_mcu_420_rgb_full_scalar_kernels(
        pixels: &[u8],
        width: u32,
        x0: u32,
        y0: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        out: &mut [[i16; 64]],
    ) {
        unsafe {
            quantize_mcu_420_rgb_full_inner::<false>(
                pixels, width, x0, y0, div_luma, div_chroma, out,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn quantize_mcu_420_rgb_full_inner<const AVX2: bool>(
        pixels: &[u8],
        width: u32,
        x0: u32,
        y0: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        out: &mut [[i16; 64]],
    ) {
        debug_assert!(x0 + 16 <= width);
        debug_assert!(out.len() >= 6);

        let stride = width as usize * RGB.bpp;
        let mut y_row = [0u8; 16];
        let mut cb_full = [0u8; 16 * 16];
        let mut cr_full = [0u8; 16 * 16];

        for j in 0..16usize {
            let row_off = (y0 as usize + j) * stride;
            let start = row_off + x0 as usize * RGB.bpp;
            let src = &pixels[start..start + 16 * RGB.bpp];
            let off = j * 16;
            let block_row = j & 7;
            let top = j < 8;
            let dst_off = block_row * 8;
            if AVX2 {
                unsafe {
                    let left_idx = if top { 0 } else { 2 };
                    let right_idx = left_idx + 1;
                    let y_left = (*out.as_mut_ptr().add(left_idx)).as_mut_ptr().add(dst_off);
                    let y_right = (*out.as_mut_ptr().add(right_idx)).as_mut_ptr().add(dst_off);
                    super::color::rgb24_16_avx2_to_luma_blocks(
                        src.as_ptr(),
                        RGB,
                        y_left,
                        y_right,
                        cb_full[off..].as_mut_ptr(),
                        cr_full[off..].as_mut_ptr(),
                    );
                }
            } else {
                super::color::rgb_row_to_ycc(
                    src,
                    RGB,
                    16,
                    &mut y_row,
                    &mut cb_full[off..off + 16],
                    &mut cr_full[off..off + 16],
                );
                let dst_left = if top { &mut out[0] } else { &mut out[2] };
                for i in 0..8 {
                    dst_left[dst_off + i] = y_row[i] as i16 - 128;
                }

                let dst_right = if top { &mut out[1] } else { &mut out[3] };
                for i in 0..8 {
                    dst_right[dst_off + i] = y_row[8 + i] as i16 - 128;
                }
            }
        }

        let mut cb_blk = [0i16; 64];
        let mut cr_blk = [0i16; 64];
        super::color::h2v2_downsample(&cb_full, &mut cb_blk);
        super::color::h2v2_downsample(&cr_full, &mut cr_blk);

        for block in &mut out[..4] {
            fdct_quantize_zigzag::<AVX2>(block, div_luma);
        }
        fdct_quantize_zigzag_into::<AVX2>(&mut cb_blk, div_chroma, &mut out[4]);
        fdct_quantize_zigzag_into::<AVX2>(&mut cr_blk, div_chroma, &mut out[5]);
    }

    fn fdct_quantize_zigzag<const AVX2: bool>(block: &mut [i16; 64], div: &Divisors) {
        let mut natural = [0i16; 64];
        if AVX2 {
            unsafe {
                super::dct::fdct_avx2(block);
                super::quant::quantize_avx2(block, div, &mut natural);
            }
        } else {
            super::dct::fdct_islow(block);
            super::quant::quantize_natural(block, div, &mut natural);
        }
        super::quant::zigzag_scatter(&natural, block);
    }

    fn fdct_quantize_zigzag_into<const AVX2: bool>(
        block: &mut [i16; 64],
        div: &Divisors,
        out: &mut [i16; 64],
    ) {
        let mut natural = [0i16; 64];
        if AVX2 {
            unsafe {
                super::dct::fdct_avx2(block);
                super::quant::quantize_avx2(block, div, &mut natural);
            }
        } else {
            super::dct::fdct_islow(block);
            super::quant::quantize_natural(block, div, &mut natural);
        }
        super::quant::zigzag_scatter(&natural, out);
    }
}

// ===========================================================================
// color: AVX2 RGBA→YCbCr per-row converter (n=16 only).
// Translated from `simd/x86_64/jccolext-avx2.asm` math, reshaped for our
// "one row at a time, 16 pixels wide" API. Other widths / RGB-bpp paths
// delegate to scalar.
// ===========================================================================
pub mod color {
    use core::arch::x86_64::*;

    use crate::color::PixelLayout;

    #[repr(C, align(32))]
    struct Aligned<T>(T);

    // F_0_337 = F_0_587 - F_0_250 = 38470 - 16384 = 22086.
    // PW_F0299_F0337 — Y: pair (R, G) with [F_0_299, F_0_337].
    static PW_F0299_F0337: Aligned<[i16; 16]> = Aligned([
        19595, 22086, 19595, 22086, 19595, 22086, 19595, 22086, 19595, 22086, 19595, 22086, 19595,
        22086, 19595, 22086,
    ]);

    // PW_F0114_F0250 — Y: pair (B, G) with [F_0_114, F_0_250].
    static PW_F0114_F0250: Aligned<[i16; 16]> = Aligned([
        7471, 16384, 7471, 16384, 7471, 16384, 7471, 16384, 7471, 16384, 7471, 16384, 7471, 16384,
        7471, 16384,
    ]);

    // PW_MF016_MF033 — Cb: pair (R, G) with [-F_0_168, -F_0_331].
    static PW_MF016_MF033: Aligned<[i16; 16]> = Aligned([
        -11059, -21709, -11059, -21709, -11059, -21709, -11059, -21709, -11059, -21709, -11059,
        -21709, -11059, -21709, -11059, -21709,
    ]);

    // PW_MF008_MF041 — Cr: pair (B, G) with [-F_0_081, -F_0_418].
    static PW_MF008_MF041: Aligned<[i16; 16]> = Aligned([
        -5329, -27439, -5329, -27439, -5329, -27439, -5329, -27439, -5329, -27439, -5329, -27439,
        -5329, -27439, -5329, -27439,
    ]);

    // PD_ONEHALF — Y rounding bias = 1 << 15.
    static PD_ONEHALF: Aligned<[i32; 8]> = Aligned([32768; 8]);

    // PD_ONEHALFM1_CJ — Cb/Cr rounding + center-128 bias.
    //   = (1 << 15) - 1 + (128 << 16) = 32767 + 8388608 = 8421375
    static PD_ONEHALFM1_CJ: Aligned<[i32; 8]> = Aligned([8421375; 8]);

    /// Per-row RGB(A) → YCbCr converter.
    ///
    /// AVX2 fast path: `n == 16` for both 4-byte layouts (RGBA / BGRA /
    /// ARGB / ABGR / RGBX / BGRX) and 3-byte layouts (RGB / BGR). All
    /// other widths fall through to the scalar reference.
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
        if n == 16 && std::arch::is_x86_feature_detected!("avx2") {
            if layout.bpp == 4 {
                unsafe {
                    rgba16_avx2(
                        pixels.as_ptr(),
                        layout,
                        y.as_mut_ptr(),
                        cb.as_mut_ptr(),
                        cr.as_mut_ptr(),
                    )
                }
            } else if layout.bpp == 3 {
                unsafe {
                    rgb24_16_avx2(
                        pixels.as_ptr(),
                        layout,
                        y.as_mut_ptr(),
                        cb.as_mut_ptr(),
                        cr.as_mut_ptr(),
                    )
                }
            } else {
                crate::arch::scalar::color::rgb_row_to_ycc(pixels, layout, n, y, cb, cr)
            }
        } else {
            crate::arch::scalar::color::rgb_row_to_ycc(pixels, layout, n, y, cb, cr)
        }
    }

    /// 16x16 → 8x8 chroma 4:2:0 box-average with libjpeg-turbo's biased
    /// rounding (`{1, 2, 1, 2, ...}` per output column). Bit-exact
    /// equivalent to `arch::scalar::color::h2v2_downsample`.
    pub fn h2v2_downsample(src: &[u8; 256], dst: &mut [i16; 64]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { h2v2_avx2(src, dst) }
        } else {
            crate::arch::scalar::color::h2v2_downsample(src, dst)
        }
    }

    /// 16x8 → 8x8 horizontal 2:1 chroma downsample with libjpeg's
    /// `{0, 1, 0, 1, ...}` bias. Bit-exact equivalent to
    /// `arch::scalar::color::h2v1_downsample`.
    pub fn h2v1_downsample(src: &[u8; 128], dst: &mut [i16; 64]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { h2v1_avx2(src, dst) }
        } else {
            crate::arch::scalar::color::h2v1_downsample(src, dst)
        }
    }

    /// # Safety
    /// AVX2 must be available (the runtime gate in
    /// `h2v1_downsample` checks). `src` / `dst` are fixed-size refs.
    #[target_feature(enable = "avx2")]
    unsafe fn h2v1_avx2(src: &[u8; 128], dst: &mut [i16; 64]) {
        unsafe {
            // Bias `{0, 1, 0, 1, ...}` over 8 u16 lanes per 128-bit
            // half, = u32 lanes of 0x0001_0000 broadcast.
            let bias = _mm256_set1_epi32(0x0001_0000u32 as i32);
            let level_shift = _mm256_set1_epi16(128);
            // `_mm256_maddubs_epi16(a, ones)` sums adjacent byte pairs
            // within each 128-bit lane: `out[i] = a[2i] + a[2i+1]`
            // (saturating to i16, but the u8+u8 sum fits comfortably).
            let ones = _mm256_set1_epi8(1);

            // Each iteration consumes 2 input rows = 32 bytes, emits
            // 2 output rows = 32 bytes (16 i16).
            for j in 0..4 {
                let row_off = j * 2 * 16;
                let r = _mm256_loadu_si256(src.as_ptr().add(row_off) as *const __m256i);
                // Pair-add within each 128-bit lane:
                //   lo 128 = pair sums for row 2j   (8 i16)
                //   hi 128 = pair sums for row 2j+1 (8 i16)
                let s = _mm256_maddubs_epi16(r, ones);
                // + bias, then /2.
                let avg = _mm256_srli_epi16::<1>(_mm256_add_epi16(s, bias));
                // Level-shift to centered i16 range.
                let signed = _mm256_sub_epi16(avg, level_shift);
                _mm256_storeu_si256(dst.as_mut_ptr().add(j * 16) as *mut __m256i, signed);
            }
            _mm256_zeroupper();
        }
    }

    /// # Safety
    /// AVX2 must be available (the runtime gate in
    /// `h2v2_downsample` checks). `src` / `dst` are fixed-size refs.
    #[target_feature(enable = "avx2")]
    unsafe fn h2v2_avx2(src: &[u8; 256], dst: &mut [i16; 64]) {
        unsafe {
            // bias `{1, 2, 1, 2, ...}` = u32 lanes of 0x0002_0001
            // broadcast — same encoding NEON uses, gives +1 on even
            // output columns and +2 on odd.
            let bias = _mm256_set1_epi32(0x0002_0001u32 as i32);
            let level_shift = _mm256_set1_epi16(128);
            // `_mm256_maddubs_epi16(a, ones)` sums adjacent byte pairs:
            // `result_lane[i] = a[2i] + a[2i+1]` (saturating to i16, but
            // the sum of two u8 fits comfortably).
            let ones = _mm256_set1_epi8(1);

            // Each iteration consumes 4 input rows = 64 bytes, emits
            // 2 output rows = 32 bytes.
            for j in 0..4 {
                let row_off = j * 4 * 16;
                let r01 = _mm256_loadu_si256(src.as_ptr().add(row_off) as *const __m256i);
                let r23 = _mm256_loadu_si256(src.as_ptr().add(row_off + 32) as *const __m256i);

                // Pairwise horizontal byte sums per 128-bit lane:
                //   s01 = [pair sums of input row 4j+0, pair sums of 4j+1] (each 8 i16)
                //   s23 = [pair sums of 4j+2, pair sums of 4j+3]
                let s01 = _mm256_maddubs_epi16(r01, ones);
                let s23 = _mm256_maddubs_epi16(r23, ones);

                // Cross-pair the lo/hi halves so each ymm holds, in its
                // two 128-bit lanes, the 8 pair-sums for output rows
                // (2j, 2j+1):
                //   lo_halves = [s01_lo, s23_lo] = [row 4j+0, row 4j+2]
                //   hi_halves = [s01_hi, s23_hi] = [row 4j+1, row 4j+3]
                let lo_halves = _mm256_permute2x128_si256::<0x20>(s01, s23);
                let hi_halves = _mm256_permute2x128_si256::<0x31>(s01, s23);

                // Vertical sum + bias = 2x2 box sums.
                let sums = _mm256_add_epi16(_mm256_add_epi16(lo_halves, hi_halves), bias);
                // /4 (avg).
                let avg = _mm256_srli_epi16::<2>(sums);
                // Level-shift to centered i16 range.
                let signed = _mm256_sub_epi16(avg, level_shift);

                // Store 16 i16 = 32 bytes = 2 output rows.
                _mm256_storeu_si256(dst.as_mut_ptr().add(j * 16) as *mut __m256i, signed);
            }
            _mm256_zeroupper();
        }
    }

    /// Deinterleave 16 4-byte pixels into three `__m256i` registers
    /// carrying 16 u16 lanes of R, G, B respectively. The R/G/B byte
    /// positions within each pixel come from `layout`, so the same
    /// kernel covers every bpp=4 ordering (RGBA / BGRA / ARGB / ABGR /
    /// RGBX / BGRX).
    ///
    /// Approach: vpshufb each input ymm to gather R/G/B/A bytes into
    /// 4-byte groups within each 128-bit lane, then vpermd across both
    /// ymm to produce 16 contiguous bytes per channel in lo 128, then
    /// vpmovzxbw to widen to 16 u16 lanes.
    ///
    /// # Safety
    /// Caller must have AVX2 enabled. Relies on inlining into a
    /// `#[target_feature(enable = "avx2")]` function. Inputs are
    /// by-value vector lanes; no memory access.
    #[inline(always)]
    unsafe fn deinterleave_pixels16(
        p0: __m256i,
        p1: __m256i,
        layout: PixelLayout,
    ) -> (__m256i, __m256i, __m256i) {
        unsafe {
            // R/G/B/A offsets sum to 0+1+2+3 = 6 for any bpp=4 layout;
            // the leftover slot is the alpha/pad byte we want to drop.
            let r = layout.r_off as i8;
            let g = layout.g_off as i8;
            let b = layout.b_off as i8;
            let a = 6 - r - g - b; // 0..=3, the alpha/pad slot.
            // Per 128-bit lane, place RRRRGGGGBBBBAAAA at bytes 0..15 by
            // gathering source bytes at offsets {r, g, b, a} stepped by 4
            // (one per pixel within the 4-pixel-per-half-lane chunk).
            #[rustfmt::skip]
            let shuf = _mm256_setr_epi8(
                r, r + 4, r + 8, r + 12,  g, g + 4, g + 8, g + 12,
                b, b + 4, b + 8, b + 12,  a, a + 4, a + 8, a + 12,
                r, r + 4, r + 8, r + 12,  g, g + 4, g + 8, g + 12,
                b, b + 4, b + 8, b + 12,  a, a + 4, a + 8, a + 12,
            );
            // After vpshufb (per-lane):
            //   s0 lo lane = [R0..3 G0..3 B0..3 A0..3]
            //   s0 hi lane = [R4..7 G4..7 B4..7 A4..7]
            //   s1 lo lane = [R8..11 G8..11 B8..11 A8..11]
            //   s1 hi lane = [R12..15 G12..15 B12..15 A12..15]
            let s0 = _mm256_shuffle_epi8(p0, shuf);
            let s1 = _mm256_shuffle_epi8(p1, shuf);

            // unpacklo/hi 32-bit chunks: align R/G chunks (resp. B/A) across
            // both inputs.
            //   lo32 lane content (per 32-bit lane): [R0..3, R8..11, G0..3, G8..11,
            //                                          R4..7, R12..15, G4..7, G12..15]
            //   hi32 lane content: same shape but with B/A.
            let lo32 = _mm256_unpacklo_epi32(s0, s1);
            let hi32 = _mm256_unpackhi_epi32(s0, s1);

            // vpermd indices to gather 16 contiguous channel bytes into lo
            // 128 of result (i.e. into 4 32-bit lanes).
            //   R: 32-bit lanes 0, 4, 1, 5 of lo32 → R0..3, R4..7, R8..11, R12..15
            //   G: lanes 2, 6, 3, 7 of lo32
            //   B: lanes 0, 4, 1, 5 of hi32
            let idx_r = _mm256_setr_epi32(0, 4, 1, 5, 0, 0, 0, 0);
            let idx_g = _mm256_setr_epi32(2, 6, 3, 7, 0, 0, 0, 0);
            let idx_b = _mm256_setr_epi32(0, 4, 1, 5, 0, 0, 0, 0);

            let r_packed = _mm256_permutevar8x32_epi32(lo32, idx_r);
            let g_packed = _mm256_permutevar8x32_epi32(lo32, idx_g);
            let b_packed = _mm256_permutevar8x32_epi32(hi32, idx_b);

            // Widen 16 u8 (in lo 128) → 16 u16 (full ymm).
            let r_u16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_packed));
            let g_u16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_packed));
            let b_u16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_packed));

            (r_u16, g_u16, b_u16)
        }
    }

    /// Compute one of Y/Cb/Cr from the deinterleaved channels. Returns a
    /// `__m256i` whose 16 u16 lanes are the output samples in order.
    ///
    /// `c_xy` is the constant ymm to pair the *primary* two channels
    /// `(x, y)` against (e.g. `[F_0_299, F_0_337]` interleaved for Y's
    /// `(R, G)` term). `extra_lo` and `extra_hi` are i32 add-ins for
    /// the residual term — for Y this is the `(B, G) madd` partial; for
    /// Cb/Cr it's `0.5 * B` (resp. `0.5 * R`) computed via the
    /// `vpunpcklwd(0, X) >> 1` trick.
    ///
    /// # Safety
    /// Caller must have AVX2 enabled. By-value vector inputs only.
    #[inline(always)]
    unsafe fn finalize_component(
        rg_or_bg_lo: __m256i,
        rg_or_bg_hi: __m256i,
        c_xy: __m256i,
        extra_lo: __m256i,
        extra_hi: __m256i,
        bias: __m256i,
    ) -> __m256i {
        unsafe {
            let p_lo = _mm256_madd_epi16(rg_or_bg_lo, c_xy);
            let p_hi = _mm256_madd_epi16(rg_or_bg_hi, c_xy);
            let s_lo = _mm256_add_epi32(_mm256_add_epi32(p_lo, extra_lo), bias);
            let s_hi = _mm256_add_epi32(_mm256_add_epi32(p_hi, extra_hi), bias);
            // Logical >> 16: the sum is always non-negative for the chosen
            // biases (Y's bias is +0.5, Cb/Cr add the +128 center plus the
            // round bias), so srli gives the same result as srai.
            let q_lo = _mm256_srli_epi32::<16>(s_lo);
            let q_hi = _mm256_srli_epi32::<16>(s_hi);
            // Pack i32 → u16 (saturating). The result interleaves lanes
            // per 128-bit lane of pack: lo 128 = [q_lo[0..3], q_hi[0..3]] =
            // pixels 0..7; hi 128 = [q_lo[4..7], q_hi[4..7]] = pixels 8..15.
            _mm256_packus_epi32(q_lo, q_hi)
        }
    }

    /// Pack 16 u16 (in one ymm) → 16 u8 in low 128 bits, then store.
    ///
    /// # Safety
    /// - Caller must have AVX2 enabled.
    /// - `out` must be writable for at least 16 bytes.
    #[inline(always)]
    unsafe fn pack_and_store_u16x16(v: __m256i, out: *mut u8) {
        unsafe {
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256::<1>(v);
            let packed = _mm_packus_epi16(lo, hi);
            _mm_storeu_si128(out as *mut __m128i, packed);
        }
    }

    /// # Safety
    /// - AVX2 must be available (the runtime gate in `rgb_row_to_ycc`
    ///   checks; `target_feature` enforces the compile-time half).
    /// - `pixels` must be readable for at least 64 bytes (16 4-byte pixels).
    /// - `y`, `cb`, `cr` must each be writable for at least 16 bytes.
    /// - `layout.bpp` must be 4.
    #[target_feature(enable = "avx2")]
    unsafe fn rgba16_avx2(
        pixels: *const u8,
        layout: PixelLayout,
        y: *mut u8,
        cb: *mut u8,
        cr: *mut u8,
    ) {
        unsafe {
            // Load 64 bytes = 16 4-byte pixels into 2 ymm.
            let p_in = pixels as *const __m256i;
            let p0 = _mm256_loadu_si256(p_in);
            let p1 = _mm256_loadu_si256(p_in.add(1));

            let (r_u16, g_u16, b_u16) = deinterleave_pixels16(p0, p1, layout);

            ycc_from_rgb16(r_u16, g_u16, b_u16, y, cb, cr);

            _mm256_zeroupper();
        }
    }

    /// Shared post-deinterleave math: given 16 R/G/B samples (each a
    /// `__m256i` of 16 u16 lanes), compute Y/Cb/Cr and store 16 u8 each.
    /// Used by both the 4-byte (`rgba16_avx2`) and 3-byte (`rgb24_16_avx2`)
    /// paths so the color transform lives in exactly one place.
    ///
    /// # Safety
    /// - Caller must have AVX2 enabled (relies on inlining into a
    ///   `#[target_feature(enable = "avx2")]` function).
    /// - `y`, `cb`, `cr` must each be writable for at least 16 bytes.
    #[inline(always)]
    unsafe fn ycc_from_rgb16(
        r_u16: __m256i,
        g_u16: __m256i,
        b_u16: __m256i,
        y: *mut u8,
        cb: *mut u8,
        cr: *mut u8,
    ) {
        unsafe {
            let (y_u16, cb_u16, cr_u16) = compute_ycc_from_rgb16(r_u16, g_u16, b_u16);
            pack_and_store_u16x16(y_u16, y);
            pack_and_store_u16x16(cb_u16, cb);
            pack_and_store_u16x16(cr_u16, cr);
        }
    }

    #[inline(always)]
    unsafe fn ycc_from_rgb16_to_luma_blocks(
        r_u16: __m256i,
        g_u16: __m256i,
        b_u16: __m256i,
        y_left: *mut i16,
        y_right: *mut i16,
        cb: *mut u8,
        cr: *mut u8,
    ) {
        unsafe {
            let (y_u16, cb_u16, cr_u16) = compute_ycc_from_rgb16(r_u16, g_u16, b_u16);
            let y_signed = _mm256_sub_epi16(y_u16, _mm256_set1_epi16(128));
            _mm_storeu_si128(y_left as *mut __m128i, _mm256_castsi256_si128(y_signed));
            _mm_storeu_si128(
                y_right as *mut __m128i,
                _mm256_extracti128_si256::<1>(y_signed),
            );

            pack_and_store_u16x16(cb_u16, cb);
            pack_and_store_u16x16(cr_u16, cr);
        }
    }

    #[inline(always)]
    unsafe fn compute_ycc_from_rgb16(
        r_u16: __m256i,
        g_u16: __m256i,
        b_u16: __m256i,
    ) -> (__m256i, __m256i, __m256i) {
        unsafe {
            let rg_lo = _mm256_unpacklo_epi16(r_u16, g_u16);
            let rg_hi = _mm256_unpackhi_epi16(r_u16, g_u16);
            let bg_lo = _mm256_unpacklo_epi16(b_u16, g_u16);
            let bg_hi = _mm256_unpackhi_epi16(b_u16, g_u16);

            // 0.5 * B and 0.5 * R via the (zero,X) interleave + >>1
            // trick: unpacklo_wd(0, B) places each B[i] in the high 16
            // bits of a u32 lane (= B[i] << 16); then srli 1 gives
            // B[i] << 15 = B[i] * 32768.
            let zero = _mm256_setzero_si256();
            let half_b_lo = _mm256_srli_epi32::<1>(_mm256_unpacklo_epi16(zero, b_u16));
            let half_b_hi = _mm256_srli_epi32::<1>(_mm256_unpackhi_epi16(zero, b_u16));
            let half_r_lo = _mm256_srli_epi32::<1>(_mm256_unpacklo_epi16(zero, r_u16));
            let half_r_hi = _mm256_srli_epi32::<1>(_mm256_unpackhi_epi16(zero, r_u16));

            let c_y_rg = _mm256_loadu_si256(PW_F0299_F0337.0.as_ptr() as *const __m256i);
            let c_y_bg = _mm256_loadu_si256(PW_F0114_F0250.0.as_ptr() as *const __m256i);
            let c_cb_rg = _mm256_loadu_si256(PW_MF016_MF033.0.as_ptr() as *const __m256i);
            let c_cr_bg = _mm256_loadu_si256(PW_MF008_MF041.0.as_ptr() as *const __m256i);
            let bias_y = _mm256_loadu_si256(PD_ONEHALF.0.as_ptr() as *const __m256i);
            let bias_cbcr = _mm256_loadu_si256(PD_ONEHALFM1_CJ.0.as_ptr() as *const __m256i);

            let y_extra_lo = _mm256_madd_epi16(bg_lo, c_y_bg);
            let y_extra_hi = _mm256_madd_epi16(bg_hi, c_y_bg);
            let y_u16 = finalize_component(rg_lo, rg_hi, c_y_rg, y_extra_lo, y_extra_hi, bias_y);
            let cb_u16 = finalize_component(rg_lo, rg_hi, c_cb_rg, half_b_lo, half_b_hi, bias_cbcr);
            let cr_u16 = finalize_component(bg_lo, bg_hi, c_cr_bg, half_r_lo, half_r_hi, bias_cbcr);
            (y_u16, cb_u16, cr_u16)
        }
    }

    /// # Safety
    /// - AVX2 must be available (the runtime gate in `rgb_row_to_ycc`
    ///   checks; `target_feature` enforces the compile-time half).
    /// - `pixels` must be readable for at least 48 bytes (16 3-byte pixels).
    /// - `y`, `cb`, `cr` must each be writable for at least 16 bytes.
    /// - `layout.bpp` must be 3.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn rgb24_16_avx2(
        pixels: *const u8,
        layout: PixelLayout,
        y: *mut u8,
        cb: *mut u8,
        cr: *mut u8,
    ) {
        unsafe {
            // Expand 16 packed RGB24 pixels (48 bytes) into two ymm in
            // 4-byte RGBA order (pad byte = 0). Each 128-bit lane holds
            // 4 pixels; the per-lane shuffle maps 12 input bytes → 16
            // output bytes, inserting a zero pad after every triple.
            // 0x80 in a vpshufb index zeroes the destination byte.
            #[rustfmt::skip]
            let m_a = _mm_setr_epi8(
                0, 1, 2, -128, 3, 4, 5, -128,
                6, 7, 8, -128, 9, 10, 11, -128,
            );
            #[rustfmt::skip]
            let m_b = _mm_setr_epi8(
                4, 5, 6, -128, 7, 8, 9, -128,
                10, 11, 12, -128, 13, 14, 15, -128,
            );

            // p0 = pixels 0..7. lane0 ← src bytes 0..11 (pixels 0..3),
            // lane1 ← src bytes 12..23 (pixels 4..7).
            let lo0 = _mm_loadu_si128(pixels as *const __m128i); // src 0..15
            let hi0 = _mm_loadu_si128(pixels.add(12) as *const __m128i); // src 12..27
            let v0 = _mm256_set_m128i(hi0, lo0);
            let p0 = _mm256_shuffle_epi8(v0, _mm256_set_m128i(m_a, m_a));

            // p1 = pixels 8..15. lo1 starts at src 24, hi1 at src 32.
            // lane0 ← src 24..35 (pixels 8..11) via M_A; lane1 ←
            // src 36..47 (pixels 12..15), which sit at bytes 4..15 of
            // hi1 (base 32), via M_B.
            let lo1 = _mm_loadu_si128(pixels.add(24) as *const __m128i); // src 24..39
            let hi1 = _mm_loadu_si128(pixels.add(32) as *const __m128i); // src 32..47
            let v1 = _mm256_set_m128i(hi1, lo1);
            let p1 = _mm256_shuffle_epi8(v1, _mm256_set_m128i(m_b, m_a));

            // Pass the original 3-byte layout: deinterleave_pixels16 reads
            // only r_off/g_off/b_off (0/1/2 for RGB, 2/1/0 for BGR) and
            // derives the pad slot a = 6 - r - g - b = 3, which is the
            // zero pad we inserted and gets dropped.
            let (r_u16, g_u16, b_u16) = deinterleave_pixels16(p0, p1, layout);

            ycc_from_rgb16(r_u16, g_u16, b_u16, y, cb, cr);

            _mm256_zeroupper();
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn rgb24_16_avx2_to_luma_blocks(
        pixels: *const u8,
        layout: PixelLayout,
        y_left: *mut i16,
        y_right: *mut i16,
        cb: *mut u8,
        cr: *mut u8,
    ) {
        unsafe {
            #[rustfmt::skip]
            let m_a = _mm_setr_epi8(
                0, 1, 2, -128, 3, 4, 5, -128,
                6, 7, 8, -128, 9, 10, 11, -128,
            );
            #[rustfmt::skip]
            let m_b = _mm_setr_epi8(
                4, 5, 6, -128, 7, 8, 9, -128,
                10, 11, 12, -128, 13, 14, 15, -128,
            );

            let lo0 = _mm_loadu_si128(pixels as *const __m128i);
            let hi0 = _mm_loadu_si128(pixels.add(12) as *const __m128i);
            let v0 = _mm256_set_m128i(hi0, lo0);
            let p0 = _mm256_shuffle_epi8(v0, _mm256_set_m128i(m_a, m_a));

            let lo1 = _mm_loadu_si128(pixels.add(24) as *const __m128i);
            let hi1 = _mm_loadu_si128(pixels.add(32) as *const __m128i);
            let v1 = _mm256_set_m128i(hi1, lo1);
            let p1 = _mm256_shuffle_epi8(v1, _mm256_set_m128i(m_b, m_a));

            let (r_u16, g_u16, b_u16) = deinterleave_pixels16(p0, p1, layout);
            ycc_from_rgb16_to_luma_blocks(r_u16, g_u16, b_u16, y_left, y_right, cb, cr);

            _mm256_zeroupper();
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn rgb24_16_avx2_to_luma_chroma_vectors(
        pixels: *const u8,
        layout: PixelLayout,
        y_left: *mut i16,
        y_right: *mut i16,
    ) -> (__m256i, __m256i) {
        unsafe {
            #[rustfmt::skip]
            let m_a = _mm_setr_epi8(
                0, 1, 2, -128, 3, 4, 5, -128,
                6, 7, 8, -128, 9, 10, 11, -128,
            );
            #[rustfmt::skip]
            let m_b = _mm_setr_epi8(
                4, 5, 6, -128, 7, 8, 9, -128,
                10, 11, 12, -128, 13, 14, 15, -128,
            );

            let lo0 = _mm_loadu_si128(pixels as *const __m128i);
            let hi0 = _mm_loadu_si128(pixels.add(12) as *const __m128i);
            let v0 = _mm256_set_m128i(hi0, lo0);
            let p0 = _mm256_shuffle_epi8(v0, _mm256_set_m128i(m_a, m_a));

            let lo1 = _mm_loadu_si128(pixels.add(24) as *const __m128i);
            let hi1 = _mm_loadu_si128(pixels.add(32) as *const __m128i);
            let v1 = _mm256_set_m128i(hi1, lo1);
            let p1 = _mm256_shuffle_epi8(v1, _mm256_set_m128i(m_b, m_a));

            let (r_u16, g_u16, b_u16) = deinterleave_pixels16(p0, p1, layout);
            let (y_u16, cb_u16, cr_u16) = compute_ycc_from_rgb16(r_u16, g_u16, b_u16);
            let y_signed = _mm256_sub_epi16(y_u16, _mm256_set1_epi16(128));
            _mm_storeu_si128(y_left as *mut __m128i, _mm256_castsi256_si128(y_signed));
            _mm_storeu_si128(
                y_right as *mut __m128i,
                _mm256_extracti128_si256::<1>(y_signed),
            );

            (cb_u16, cr_u16)
        }
    }

    // ---- Decoder-side kernels (currently scalar; AVX2 ports pending) ----

    /// 8x8 → 16x16 box-upsample (decoder-side counterpart of `h2v2_downsample`).
    pub fn h2v2_upsample(src: &[u8; 64], dst: &mut [u8; 256]) {
        crate::arch::scalar::color::h2v2_upsample(src, dst)
    }

    /// 8x8 → 16x8 box-upsample (decoder-side counterpart of `h2v1_downsample`).
    pub fn h2v1_upsample(src: &[u8; 64], dst: &mut [u8; 128]) {
        crate::arch::scalar::color::h2v1_upsample(src, dst)
    }

    /// Per-row YCbCr → RGB(A) converter.
    ///
    /// AVX2 fast path: processes 16-pixel chunks via `ycc_block16_avx2`;
    /// the final `n % 16` pixels and the entire request on non-AVX2 CPUs
    /// fall through to the scalar reference. Bit-identical to
    /// `arch::scalar::color::ycc_row_to_rgb` on every input.
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
        if n >= 16 && std::arch::is_x86_feature_detected!("avx2") {
            let chunks = n / 16;
            let bulk = chunks * 16;
            unsafe {
                ycc_bulk_avx2(
                    y.as_ptr(),
                    cb.as_ptr(),
                    cr.as_ptr(),
                    out.as_mut_ptr(),
                    chunks,
                    layout,
                );
            }
            let tail = n - bulk;
            if tail > 0 {
                crate::arch::scalar::color::ycc_row_to_rgb(
                    &y[bulk..],
                    &cb[bulk..],
                    &cr[bulk..],
                    &mut out[bulk * layout.bpp..],
                    tail,
                    layout,
                );
            }
        } else {
            crate::arch::scalar::color::ycc_row_to_rgb(y, cb, cr, out, n, layout)
        }
    }

    /// # Safety
    /// - AVX2 must be available (caller is gated by `is_x86_feature_detected`).
    /// - `y`, `cb`, `cr` must each be readable for at least `chunks * 16` bytes.
    /// - `out` must be writable for at least `chunks * 16 * layout.bpp` bytes.
    #[target_feature(enable = "avx2")]
    unsafe fn ycc_bulk_avx2(
        y: *const u8,
        cb: *const u8,
        cr: *const u8,
        out: *mut u8,
        chunks: usize,
        layout: PixelLayout,
    ) {
        unsafe {
            let bpp = layout.bpp;
            for k in 0..chunks {
                let (r, g, b) = compute_rgb_block16(y.add(k * 16), cb.add(k * 16), cr.add(k * 16));
                let dst = out.add(k * 16 * bpp);
                if bpp == 4 {
                    store_block16_bpp4(r, g, b, dst, layout);
                } else {
                    store_block16_bpp3(r, g, b, dst, layout);
                }
            }
            _mm256_zeroupper();
        }
    }

    /// Inverse YCbCr → RGB for 16 contiguous pixels.
    ///
    /// Computes per-pixel:
    /// ```text
    ///   R = clamp(Y + ((I_CR_R * (Cr-128) + HALF) >> 16), 0, 255)
    ///   G = clamp(Y - ((I_CB_G * (Cb-128) + I_CR_G * (Cr-128) + HALF) >> 16), 0, 255)
    ///   B = clamp(Y + ((I_CB_B * (Cb-128) + HALF) >> 16), 0, 255)
    /// ```
    /// in i32 arithmetic, then saturating-packs to u8. The saturating
    /// pack chain (`packs_epi32` → `packus_epi16`) reproduces the scalar
    /// `clamp(0, 255)` bit-for-bit.
    ///
    /// Returns `(R, G, B)` as three xmm registers, each holding 16 u8
    /// channel samples in natural pixel order.
    ///
    /// # Safety
    /// - AVX2 must be enabled.
    /// - Each input pointer must be readable for 16 bytes.
    #[inline(always)]
    unsafe fn compute_rgb_block16(
        y: *const u8,
        cb: *const u8,
        cr: *const u8,
    ) -> (__m128i, __m128i, __m128i) {
        unsafe {
            const I_CR_R: i32 = 91881;
            const I_CB_G: i32 = 22554;
            const I_CR_G: i32 = 46802;
            const I_CB_B: i32 = 116130;
            const HALF: i32 = 1 << 15;

            let y_i16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(y as *const __m128i));
            let cb_i16 = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128(cb as *const __m128i)),
                _mm256_set1_epi16(128),
            );
            let cr_i16 = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm_loadu_si128(cr as *const __m128i)),
                _mm256_set1_epi16(128),
            );

            // Split i16x16 into two i32x8 halves: lo lane (pixels 0..7)
            // and hi lane (pixels 8..15).
            let y_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y_i16));
            let y_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(y_i16));
            let cb_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(cb_i16));
            let cb_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(cb_i16));
            let cr_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(cr_i16));
            let cr_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(cr_i16));

            let k_cr_r = _mm256_set1_epi32(I_CR_R);
            let k_cb_g = _mm256_set1_epi32(I_CB_G);
            let k_cr_g = _mm256_set1_epi32(I_CR_G);
            let k_cb_b = _mm256_set1_epi32(I_CB_B);
            let half = _mm256_set1_epi32(HALF);

            // Cb/Cr ∈ [-128, 127] and constants fit in 18 bits, so each
            // product fits in i32 without overflow.
            let cr_r_lo =
                _mm256_srai_epi32::<16>(_mm256_add_epi32(_mm256_mullo_epi32(cr_lo, k_cr_r), half));
            let cr_r_hi =
                _mm256_srai_epi32::<16>(_mm256_add_epi32(_mm256_mullo_epi32(cr_hi, k_cr_r), half));
            let r_lo = _mm256_add_epi32(y_lo, cr_r_lo);
            let r_hi = _mm256_add_epi32(y_hi, cr_r_hi);

            let cb_b_lo =
                _mm256_srai_epi32::<16>(_mm256_add_epi32(_mm256_mullo_epi32(cb_lo, k_cb_b), half));
            let cb_b_hi =
                _mm256_srai_epi32::<16>(_mm256_add_epi32(_mm256_mullo_epi32(cb_hi, k_cb_b), half));
            let b_lo = _mm256_add_epi32(y_lo, cb_b_lo);
            let b_hi = _mm256_add_epi32(y_hi, cb_b_hi);

            let g_sub_lo = _mm256_srai_epi32::<16>(_mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cb_lo, k_cb_g),
                    _mm256_mullo_epi32(cr_lo, k_cr_g),
                ),
                half,
            ));
            let g_sub_hi = _mm256_srai_epi32::<16>(_mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cb_hi, k_cb_g),
                    _mm256_mullo_epi32(cr_hi, k_cr_g),
                ),
                half,
            ));
            let g_lo = _mm256_sub_epi32(y_lo, g_sub_lo);
            let g_hi = _mm256_sub_epi32(y_hi, g_sub_hi);

            // i32 → i16 with signed saturation. `packs_epi32(a, b)`
            // interleaves per 128-bit lane:
            //   lo128 = [a.lo4, b.lo4]   hi128 = [a.hi4, b.hi4]
            // To restore [pixels 0..7, pixels 8..15] order we permute
            // 64-bit lanes by 0xD8 = [0, 2, 1, 3].
            const PERM: i32 = 0b11_01_10_00;
            let r_i16 = _mm256_permute4x64_epi64::<PERM>(_mm256_packs_epi32(r_lo, r_hi));
            let g_i16 = _mm256_permute4x64_epi64::<PERM>(_mm256_packs_epi32(g_lo, g_hi));
            let b_i16 = _mm256_permute4x64_epi64::<PERM>(_mm256_packs_epi32(b_lo, b_hi));

            // i16 → u8 with unsigned saturation. `packus_epi16(a, a)`
            // collapses 16 i16 in one ymm into 16 u8 in the low 128
            // (per-lane: lo128 of a → bytes 0..7 in lo of each 128-bit
            // half), so we permute again.
            let r_u8 = _mm256_permute4x64_epi64::<PERM>(_mm256_packus_epi16(r_i16, r_i16));
            let g_u8 = _mm256_permute4x64_epi64::<PERM>(_mm256_packus_epi16(g_i16, g_i16));
            let b_u8 = _mm256_permute4x64_epi64::<PERM>(_mm256_packus_epi16(b_i16, b_i16));

            (
                _mm256_castsi256_si128(r_u8),
                _mm256_castsi256_si128(g_u8),
                _mm256_castsi256_si128(b_u8),
            )
        }
    }

    /// Interleave R/G/B (+ 0xFF alpha) into a 64-byte 4-bpp output for 16
    /// pixels. The channel-to-byte mapping is derived from `layout`; only
    /// the six bpp=4 layouts (RGBA / BGRA / ARGB / ABGR / RGBX / BGRX)
    /// are valid here.
    ///
    /// # Safety
    /// - AVX2 / SSE2 must be enabled.
    /// - `out` must be writable for 64 bytes.
    /// - `layout.bpp` must be 4.
    #[inline(always)]
    unsafe fn store_block16_bpp4(
        r: __m128i,
        g: __m128i,
        b: __m128i,
        out: *mut u8,
        layout: PixelLayout,
    ) {
        unsafe {
            let alpha = _mm_set1_epi8(0xFFu8 as i8);
            // Pick the channel xmm that should land at byte offsets 0/1/2
            // (and the 0xFF pad goes at the leftover offset 3 - r - g - b
            // within the source ordering). We sort r/g/b/alpha into c0..c3
            // such that pixel layout = [c0, c1, c2, c3].
            let r_off = layout.r_off;
            let g_off = layout.g_off;
            let b_off = layout.b_off;
            let a_off = 6 - r_off - g_off - b_off;
            let mut slot = [alpha; 4];
            slot[r_off] = r;
            slot[g_off] = g;
            slot[b_off] = b;
            slot[a_off] = alpha;

            // unpack[lo|hi]_epi8(c01, c23) gives, per byte position i in [0,16):
            //   lo: c01[i/2] if i even, c23[i/2] if i odd? Actually
            //   _mm_unpacklo_epi8(a, b) = [a0 b0 a1 b1 ... a7 b7]. So
            //   pairing slot[0]/slot[1] gives [c0_0 c1_0 c0_1 c1_1 ...].
            let c01_lo = _mm_unpacklo_epi8(slot[0], slot[1]);
            let c01_hi = _mm_unpackhi_epi8(slot[0], slot[1]);
            let c23_lo = _mm_unpacklo_epi8(slot[2], slot[3]);
            let c23_hi = _mm_unpackhi_epi8(slot[2], slot[3]);

            // unpack[lo|hi]_epi16 pairs the (c0,c1) bytes with (c2,c3)
            // bytes per pixel into 4-byte groups.
            let px_0_3 = _mm_unpacklo_epi16(c01_lo, c23_lo);
            let px_4_7 = _mm_unpackhi_epi16(c01_lo, c23_lo);
            let px_8_11 = _mm_unpacklo_epi16(c01_hi, c23_hi);
            let px_12_15 = _mm_unpackhi_epi16(c01_hi, c23_hi);

            _mm_storeu_si128(out as *mut __m128i, px_0_3);
            _mm_storeu_si128(out.add(16) as *mut __m128i, px_4_7);
            _mm_storeu_si128(out.add(32) as *mut __m128i, px_8_11);
            _mm_storeu_si128(out.add(48) as *mut __m128i, px_12_15);
        }
    }

    /// Interleave R/G/B into a 48-byte 3-bpp output for 16 pixels. AVX2
    /// shines on the per-channel math; the 3-byte cross-lane interleave
    /// is done by spilling each channel to a 16-byte stack buffer and
    /// scalar-storing each pixel, which is simpler than the asm-style
    /// shuffle-and-permute and still benefits from the SIMD compute.
    ///
    /// # Safety
    /// - AVX2 / SSE2 must be enabled.
    /// - `out` must be writable for 48 bytes.
    /// - `layout.bpp` must be 3 (RGB / BGR).
    #[inline(always)]
    unsafe fn store_block16_bpp3(
        r: __m128i,
        g: __m128i,
        b: __m128i,
        out: *mut u8,
        layout: PixelLayout,
    ) {
        unsafe {
            // Map r/g/b into (c0, c1, c2) = the three output channel
            // positions in interleave order (= byte 0, 1, 2 of each
            // pixel). For RGB layout: c0=r, c1=g, c2=b. For BGR: c0=b,
            // c1=g, c2=r. (layout.r_off / g_off / b_off ∈ {0,1,2} and
            // sum to 3.)
            let mut slot = [r; 3];
            slot[layout.r_off] = r;
            slot[layout.g_off] = g;
            slot[layout.b_off] = b;
            store_block16_interleaved_ssse3(slot[0], slot[1], slot[2], out);
        }
    }

    /// PSHUFB-based 3-byte interleave for 16 pixels (= 48 bytes).
    ///
    /// Given three xmm inputs `c0, c1, c2` (= the first/second/third
    /// byte of each pixel in the final layout), writes 48 contiguous
    /// bytes to `out` such that pixel `i` occupies bytes
    /// `[3*i, 3*i+3)` and pixel-byte `j` ∈ {0,1,2} is `c{j}[i]`.
    ///
    /// Replaces a 16-iter scalar interleave loop (= 48 VPEXTRBs +
    /// 48 byte stores) with 9 PSHUFB + 6 POR + 3 16-byte stores.
    /// On Cascade Lake (Skylake-SP family) this lifts the channel
    /// interleave from ~15% of `ycc_bulk_avx2` self-time down to
    /// near-zero.
    ///
    /// # Safety
    /// - SSSE3 must be available (PSHUFB). Caller is gated by AVX2,
    ///   which implies SSSE3.
    /// - `out` must be writable for 48 bytes.
    #[inline(always)]
    unsafe fn store_block16_interleaved_ssse3(c0: __m128i, c1: __m128i, c2: __m128i, out: *mut u8) {
        unsafe {
            // Shuffle masks: PSHUFB picks source byte at the index given
            // in the mask byte, or zeros the lane when the high bit is
            // set (0x80 sentinel). Each output 16-byte chunk takes 6
            // bytes from one channel and 5 from each of the other two.
            //
            // out0 = [c0_0 c1_0 c2_0 c0_1 c1_1 c2_1 ... c0_5]
            //                                       ^ byte 15
            // out1 = [c1_5 c2_5 c0_6 c1_6 c2_6 c0_7 ... c1_10]
            // out2 = [c2_10 c0_11 c1_11 c2_11 ... c1_15 c2_15]
            const M0_C0: [i8; 16] = [0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5];
            const M0_C1: [i8; 16] = [-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1];
            const M0_C2: [i8; 16] = [-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1];
            const M1_C0: [i8; 16] = [-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1];
            const M1_C1: [i8; 16] = [5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10];
            const M1_C2: [i8; 16] = [-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1];
            const M2_C0: [i8; 16] = [
                -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
            ];
            const M2_C1: [i8; 16] = [
                -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
            ];
            const M2_C2: [i8; 16] = [
                10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
            ];

            let m0_c0 = _mm_loadu_si128(M0_C0.as_ptr() as *const __m128i);
            let m0_c1 = _mm_loadu_si128(M0_C1.as_ptr() as *const __m128i);
            let m0_c2 = _mm_loadu_si128(M0_C2.as_ptr() as *const __m128i);
            let m1_c0 = _mm_loadu_si128(M1_C0.as_ptr() as *const __m128i);
            let m1_c1 = _mm_loadu_si128(M1_C1.as_ptr() as *const __m128i);
            let m1_c2 = _mm_loadu_si128(M1_C2.as_ptr() as *const __m128i);
            let m2_c0 = _mm_loadu_si128(M2_C0.as_ptr() as *const __m128i);
            let m2_c1 = _mm_loadu_si128(M2_C1.as_ptr() as *const __m128i);
            let m2_c2 = _mm_loadu_si128(M2_C2.as_ptr() as *const __m128i);

            let out0 = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(c0, m0_c0), _mm_shuffle_epi8(c1, m0_c1)),
                _mm_shuffle_epi8(c2, m0_c2),
            );
            let out1 = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(c0, m1_c0), _mm_shuffle_epi8(c1, m1_c1)),
                _mm_shuffle_epi8(c2, m1_c2),
            );
            let out2 = _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(c0, m2_c0), _mm_shuffle_epi8(c1, m2_c1)),
                _mm_shuffle_epi8(c2, m2_c2),
            );

            _mm_storeu_si128(out as *mut __m128i, out0);
            _mm_storeu_si128(out.add(16) as *mut __m128i, out1);
            _mm_storeu_si128(out.add(32) as *mut __m128i, out2);
        }
    }
}

// ===========================================================================
// quant: AVX2 reciprocal-multiply quantize, natural-order output.
// Translated from `simd/x86_64/jquanti-avx2.asm::jsimd_quantize_avx2`.
// ===========================================================================
pub mod quant {
    use core::arch::x86_64::*;

    use crate::tables::Divisors;

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

    /// Permute `natural` (DCT natural order) into `zz` (zig-zag order).
    pub fn zigzag_scatter(natural: &[i16; 64], zz: &mut [i16; 64]) {
        macro_rules! z {
            ($dst:literal, $src:literal) => {
                zz[$dst] = natural[$src];
            };
        }

        z!(0, 0);
        z!(1, 1);
        z!(2, 8);
        z!(3, 16);
        z!(4, 9);
        z!(5, 2);
        z!(6, 3);
        z!(7, 10);
        z!(8, 17);
        z!(9, 24);
        z!(10, 32);
        z!(11, 25);
        z!(12, 18);
        z!(13, 11);
        z!(14, 4);
        z!(15, 5);
        z!(16, 12);
        z!(17, 19);
        z!(18, 26);
        z!(19, 33);
        z!(20, 40);
        z!(21, 48);
        z!(22, 41);
        z!(23, 34);
        z!(24, 27);
        z!(25, 20);
        z!(26, 13);
        z!(27, 6);
        z!(28, 7);
        z!(29, 14);
        z!(30, 21);
        z!(31, 28);
        z!(32, 35);
        z!(33, 42);
        z!(34, 49);
        z!(35, 56);
        z!(36, 57);
        z!(37, 50);
        z!(38, 43);
        z!(39, 36);
        z!(40, 29);
        z!(41, 22);
        z!(42, 15);
        z!(43, 23);
        z!(44, 30);
        z!(45, 37);
        z!(46, 44);
        z!(47, 51);
        z!(48, 58);
        z!(49, 59);
        z!(50, 52);
        z!(51, 45);
        z!(52, 38);
        z!(53, 31);
        z!(54, 39);
        z!(55, 46);
        z!(56, 53);
        z!(57, 60);
        z!(58, 61);
        z!(59, 54);
        z!(60, 47);
        z!(61, 55);
        z!(62, 62);
        z!(63, 63);
    }

    /// # Safety
    /// AVX2 must be available (the runtime gate in `quantize_natural`
    /// checks). All inputs are fixed-size references.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn quantize_avx2(block: &[i16; 64], div: &Divisors, out: &mut [i16; 64]) {
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
// sample: AVX2 fancy chroma upsample (decoder), translated from
// `simd/x86_64/jdsample-avx2.asm`.
// ===========================================================================
pub mod sample {
    use core::arch::x86_64::*;

    /// Vertical pass of libjpeg-turbo's `h2v2_fancy` upsample: per chroma
    /// column, blend the current row with one of its neighbors
    /// (`out[i] = (3*cur[i] + nbr[i] + 2) >> 2`). Bit-exact equivalent to
    /// `arch::scalar::sample::h2v2_fancy_vblend`.
    pub fn h2v2_fancy_vblend(cur: &[u8], nbr: &[u8], out: &mut [u8], n: usize) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { h2v2_fancy_vblend_avx2(cur, nbr, out, n) }
        } else {
            crate::arch::scalar::sample::h2v2_fancy_vblend(cur, nbr, out, n)
        }
    }

    /// Horizontal pass of libjpeg-turbo's `h2_fancy` upsample: produce
    /// `2 * n` output samples from `n` chroma samples using the symmetric
    /// 3:1 weighted blend with `src[-1] = src[0]` / `src[n] = src[n-1]`
    /// edge clamping. Bit-exact equivalent to
    /// `arch::scalar::sample::h2_fancy_upsample`.
    pub fn h2_fancy_upsample(src: &[u8], dst: &mut [u8], n: usize) {
        // The AVX2 path needs at least one interior chunk plus the i=0
        // head sample; below 34 src lanes the scalar reference is faster
        // and avoids any edge-case bookkeeping.
        if n >= 34 && std::arch::is_x86_feature_detected!("avx2") {
            unsafe { h2_fancy_upsample_avx2(src, dst, n) }
        } else {
            crate::arch::scalar::sample::h2_fancy_upsample(src, dst, n)
        }
    }

    /// # Safety
    /// - AVX2 must be available (caller is gated by
    ///   `is_x86_feature_detected`).
    /// - `cur`, `nbr`, `out` must each be readable / writable for `n`
    ///   bytes.
    #[target_feature(enable = "avx2")]
    unsafe fn h2v2_fancy_vblend_avx2(cur: &[u8], nbr: &[u8], out: &mut [u8], n: usize) {
        unsafe {
            // `_mm256_maddubs_epi16(pairs, w)` computes, for each adjacent
            // (cur, nbr) byte pair: `cur*3 + nbr*1` (u8 × i8 → i16).
            // Encoding 0x0103 as i16 = bytes [0x03, 0x01] in
            // little-endian, so the first byte of every pair (cur)
            // multiplies by 3 and the second (nbr) by 1.
            let w = _mm256_set1_epi16(0x0103);
            let two = _mm256_set1_epi16(2);

            let mut i = 0usize;
            while i + 32 <= n {
                let c = _mm256_loadu_si256(cur.as_ptr().add(i) as *const __m256i);
                let nb = _mm256_loadu_si256(nbr.as_ptr().add(i) as *const __m256i);
                // unpack[lo|hi]_epi8 interleaves cur/nbr bytes per
                // 128-bit lane:
                //   lo lane: [c0,n0,c1,n1,...,c7,n7] (and bytes 16..23
                //   in the upper 128-bit lane); hi covers 8..15 / 24..31.
                let pairs_lo = _mm256_unpacklo_epi8(c, nb);
                let pairs_hi = _mm256_unpackhi_epi8(c, nb);
                let s_lo = _mm256_srli_epi16::<2>(_mm256_add_epi16(
                    _mm256_maddubs_epi16(pairs_lo, w),
                    two,
                ));
                let s_hi = _mm256_srli_epi16::<2>(_mm256_add_epi16(
                    _mm256_maddubs_epi16(pairs_hi, w),
                    two,
                ));
                // packus_epi16 per-lane: lo result lane = [s_lo.lo8,
                // s_hi.lo8] = bytes 0..15; hi lane = [s_lo.hi8, s_hi.hi8]
                // = bytes 16..31. Saturation is a no-op since each lane
                // already fits in u8.
                let r = _mm256_packus_epi16(s_lo, s_hi);
                _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, r);
                i += 32;
            }
            _mm256_zeroupper();
            // Scalar tail handles n % 32 lanes.
            if i < n {
                crate::arch::scalar::sample::h2v2_fancy_vblend(
                    &cur[i..],
                    &nbr[i..],
                    &mut out[i..],
                    n - i,
                );
            }
        }
    }

    /// # Safety
    /// - AVX2 must be available (caller is gated by
    ///   `is_x86_feature_detected`).
    /// - `n >= 34` (caller-enforced).
    /// - `src` must be readable for `n` bytes; `dst` writable for `2 * n`
    ///   bytes.
    #[target_feature(enable = "avx2")]
    unsafe fn h2_fancy_upsample_avx2(src: &[u8], dst: &mut [u8], n: usize) {
        unsafe {
            debug_assert!(n >= 34);
            debug_assert!(src.len() >= n);
            debug_assert!(dst.len() >= 2 * n);

            // Head sample (i = 0): prev is clamped to cur, so
            // dst[0] = (3*cur + cur + 2) >> 2 = cur.
            let cur0 = src[0];
            let next0 = src[1] as u16;
            dst[0] = cur0;
            dst[1] = ((cur0 as u16 * 3 + next0 + 2) >> 2) as u8;

            let w = _mm256_set1_epi16(0x0103);
            let two = _mm256_set1_epi16(2);

            // Interior: i in [1, n-1). Each chunk reads src[i-1..i+33]
            // (unaligned loads of cur / prev / next) and writes
            // dst[2i..2i+64]. Loop bound `i + 33 <= n` ensures the
            // `next` load and the i+31 sample's true next neighbor are
            // both in range.
            let mut i = 1usize;
            while i + 33 <= n {
                let cur = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                let prev = _mm256_loadu_si256(src.as_ptr().add(i - 1) as *const __m256i);
                let next = _mm256_loadu_si256(src.as_ptr().add(i + 1) as *const __m256i);

                let pe_lo = _mm256_unpacklo_epi8(cur, prev);
                let pe_hi = _mm256_unpackhi_epi8(cur, prev);
                let po_lo = _mm256_unpacklo_epi8(cur, next);
                let po_hi = _mm256_unpackhi_epi8(cur, next);

                let e_lo =
                    _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_maddubs_epi16(pe_lo, w), two));
                let e_hi =
                    _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_maddubs_epi16(pe_hi, w), two));
                let o_lo =
                    _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_maddubs_epi16(po_lo, w), two));
                let o_hi =
                    _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_maddubs_epi16(po_hi, w), two));

                // Pack i16→u8 per lane: even_b lane0 = even outputs for
                // src indices i..i+15, lane1 = i+16..i+31. Same for odd.
                let even_b = _mm256_packus_epi16(e_lo, e_hi);
                let odd_b = _mm256_packus_epi16(o_lo, o_hi);

                // Interleave even/odd bytes to produce final output.
                // unpacklo/hi_epi8 work per-lane, so the lo-of-lane and
                // hi-of-lane halves end up split across the two ymm.
                // permute2x128 reassembles them into linear order:
                //   out0 = [lo.lane0, hi.lane0] = bytes [e0,o0,...,e15,o15]
                //   out1 = [lo.lane1, hi.lane1] = bytes [e16,o16,...,e31,o31]
                let lo = _mm256_unpacklo_epi8(even_b, odd_b);
                let hi = _mm256_unpackhi_epi8(even_b, odd_b);
                let out0 = _mm256_permute2x128_si256::<0x20>(lo, hi);
                let out1 = _mm256_permute2x128_si256::<0x31>(lo, hi);

                _mm256_storeu_si256(dst.as_mut_ptr().add(2 * i) as *mut __m256i, out0);
                _mm256_storeu_si256(dst.as_mut_ptr().add(2 * i + 32) as *mut __m256i, out1);

                i += 32;
            }
            _mm256_zeroupper();

            // Tail scalar: i in [i, n). For i < n-1 both neighbors are
            // in-bounds; the i = n-1 case clamps next to cur.
            while i < n {
                let cur = src[i] as u16;
                let prev = src[i - 1] as u16;
                let next = if i + 1 >= n { cur } else { src[i + 1] as u16 };
                dst[2 * i] = ((cur * 3 + prev + 2) >> 2) as u8;
                dst[2 * i + 1] = ((cur * 3 + next + 2) >> 2) as u8;
                i += 1;
            }
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
        10703, 4433, 10703, 4433, 10703, 4433, 10703, 4433, -10704, 4433, -10704, 4433, -10704,
        4433, -10704, 4433,
    ]);

    // PW_MF078_F117_F078_F117:
    //   times 4 dw (F_1_175 - F_1_961),  F_1_175
    //   times 4 dw (F_1_175 - F_0_390),  F_1_175
    static PW_MF078_F117_F078_F117: Aligned16<[i16; 16]> = Aligned16([
        -6436, 9633, -6436, 9633, -6436, 9633, -6436, 9633, 6437, 9633, 6437, 9633, 6437, 9633,
        6437, 9633,
    ]);

    // PW_MF060_MF089_MF050_MF256:
    //   times 4 dw (F_0_298 - F_0_899), -F_0_899
    //   times 4 dw (F_2_053 - F_2_562), -F_2_562
    static PW_MF060_MF089_MF050_MF256: Aligned16<[i16; 16]> = Aligned16([
        -4927, -7373, -4927, -7373, -4927, -7373, -4927, -7373, -4176, -20995, -4176, -20995,
        -4176, -20995, -4176, -20995,
    ]);

    // PW_F050_MF256_F060_MF089:
    //   times 4 dw (F_3_072 - F_2_562), -F_2_562
    //   times 4 dw (F_1_501 - F_0_899), -F_0_899
    static PW_F050_MF256_F060_MF089: Aligned16<[i16; 16]> = Aligned16([
        4177, -20995, 4177, -20995, 4177, -20995, 4177, -20995, 4926, -7373, 4926, -7373, 4926,
        -7373, 4926, -7373,
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
    static PW_1_NEG1: Aligned16<[i16; 16]> =
        Aligned16([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]);

    // IDCT-only constants — translated from jidctint-avx2.asm.

    // PW_MF089_F060_MF256_F050:
    //   times 4 dw -F_0_899, (F_1_501 - F_0_899)
    //   times 4 dw -F_2_562, (F_3_072 - F_2_562)
    static PW_MF089_F060_MF256_F050: Aligned16<[i16; 16]> = Aligned16([
        -7373, 4926, -7373, 4926, -7373, 4926, -7373, 4926, -20995, 4177, -20995, 4177, -20995,
        4177, -20995, 4177,
    ]);

    // IDCT's DESCALE_P2 = CONST_BITS + PASS1_BITS + 3 = 18, so the
    // pre-shift round bias is 1 << 17 = 131072. (The FDCT uses 15 / 1<<14;
    // the IDCT's "+3" absorbs the *8 factor that pass-1 didn't undo.)
    static IDCT_PD_DESCALE_P2: Aligned16<[i32; 8]> = Aligned16([131072; 8]);

    // PB_CENTERJSAMP: 32 bytes of 0x80, added byte-wise after the final
    // i16→i8 saturating pack to realize the +128 level shift via wrap.
    static PB_CENTERJSAMP: Aligned16<[u8; 32]> = Aligned16([128; 32]);

    /// 8x8 inverse integer DCT (LL&M "islow"). Bit-exact equivalent to
    /// `arch::scalar::dct::idct_islow`. Dispatches to AVX2 when
    /// available; otherwise falls back to the scalar reference.
    pub fn idct_islow(coef: &[i16; 64], output: &mut [u8; 64]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { idct_avx2(coef, output) }
        } else {
            crate::arch::scalar::dct::idct_islow(coef, output)
        }
    }

    pub fn fdct_islow(data: &mut [i16; 64]) {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { fdct_avx2(data) }
        } else {
            crate::arch::scalar::dct::fdct_islow(data)
        }
    }

    /// # Safety
    /// - Caller must have AVX2 enabled (relies on inlining).
    /// - `p` must point to at least 32 readable bytes.
    #[inline(always)]
    unsafe fn load(p: *const i16) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }

    /// In-place 8x8x16-bit transpose. Mirrors the DOTRANSPOSE asm macro:
    /// 4 input ymm each holding two rows packed (low/high 128 bits) →
    /// 4 output ymm where each holds two columns packed.
    ///
    /// # Safety
    /// Caller must have AVX2 enabled (we rely on inlining into a
    /// `#[target_feature(enable = "avx2")]` function to satisfy the
    /// instruction-availability requirement). By-value vector inputs
    /// only.
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
    /// Returns `(data0_4, data3_1, data2_6, data7_5)`.
    ///
    /// # Safety
    /// Caller must have AVX2 enabled (relies on inlining; see
    /// `dotranspose`). By-value vector inputs only.
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

    /// # Safety
    /// AVX2 must be available (the runtime gate in `fdct_islow`
    /// checks). `data` is a fixed-size mut reference.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fdct_avx2(data: &mut [i16; 64]) {
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

    // ===========================================================================
    // IDCT — translated from jidctint-avx2.asm
    // ===========================================================================

    /// 8x8 i16 transpose used between the IDCT's pass 1 and pass 2,
    /// translated directly from the asm DOTRANSPOSE macro:
    /// `vpermq → vpunpcklwd/vpunpckhwd → vpunpcklwd/vpunpckhwd →
    ///  vpunpcklqdq/vpunpckhqdq`.
    ///
    /// Inputs: `(data0_1, data3_2, data4_5, data7_6)` packed as rows.
    /// Outputs: `(data0_4, data1_5, data2_6, data3_7)` — the transposed
    /// view, each ymm holding a row pair k / k+4 of the transposed matrix.
    ///
    /// # Safety
    /// Caller must have AVX2 enabled (relies on inlining into a
    /// `#[target_feature(enable = "avx2")]` function).
    #[inline(always)]
    unsafe fn idct_dotranspose(
        m1: __m256i,
        m2: __m256i,
        m3: __m256i,
        m4: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            // qword shuffle with imm 0xD8 / 0x72 / 0xD8 / 0x72
            let t5 = _mm256_permute4x64_epi64::<0xD8>(m1);
            let t6 = _mm256_permute4x64_epi64::<0x72>(m2);
            let t7 = _mm256_permute4x64_epi64::<0xD8>(m3);
            let t8 = _mm256_permute4x64_epi64::<0x72>(m4);

            // 16-bit lane interleave (pairs)
            let r1 = _mm256_unpacklo_epi16(t5, t6);
            let r2 = _mm256_unpackhi_epi16(t5, t6);
            let r3 = _mm256_unpacklo_epi16(t7, t8);
            let r4 = _mm256_unpackhi_epi16(t7, t8);

            // 16-bit lane interleave (quads)
            let t5 = _mm256_unpacklo_epi16(r1, r2);
            let t6 = _mm256_unpacklo_epi16(r3, r4);
            let t7 = _mm256_unpackhi_epi16(r1, r2);
            let t8 = _mm256_unpackhi_epi16(r3, r4);

            // 64-bit lane interleave (final row pairs)
            let o1 = _mm256_unpacklo_epi64(t5, t6);
            let o2 = _mm256_unpackhi_epi64(t5, t6);
            let o3 = _mm256_unpacklo_epi64(t7, t8);
            let o4 = _mm256_unpackhi_epi64(t7, t8);

            (o1, o2, o3, o4)
        }
    }

    /// One 1-D IDCT pass over 16 packed columns (held across 4 ymm).
    /// Const-generic on pass id (1 or 2) selects the descale shift
    /// (DESCALE_P1 = 11 vs DESCALE_P2 = 18) and the matching round bias.
    /// Translated from the IDCT asm DODCT macro.
    ///
    /// Inputs encode rows as: `(in0_4, in3_1, in2_6, in7_5)` —
    /// each ymm's lo/hi 128 hold a row of the input.
    /// Outputs: `(data0_1, data3_2, data4_5, data7_6)`, i16-packed.
    ///
    /// # Safety
    /// Caller must have AVX2 enabled (relies on inlining; see
    /// `idct_dotranspose`).
    #[inline(always)]
    unsafe fn idct_dodct<const PASS: i32>(
        in0_4: __m256i,
        in3_1: __m256i,
        in2_6: __m256i,
        in7_5: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            // -- Even part: tmp3_2 from cols 2, 6 ----------------------
            // pair (in2[k], in6[k]) madd (F_0_541+F_0_765, F_0_541) = tmp3
            // pair (in6[k], in2[k]) madd (F_0_541-F_1_847, F_0_541) = tmp2
            let in6_2 = _mm256_permute2x128_si256::<0x01>(in2_6, in2_6);
            let l = _mm256_unpacklo_epi16(in2_6, in6_2);
            let h = _mm256_unpackhi_epi16(in2_6, in6_2);
            let pw_f130 = load(PW_F130_F054_MF130_F054.0.as_ptr());
            let tmp3_2_l = _mm256_madd_epi16(l, pw_f130);
            let tmp3_2_h = _mm256_madd_epi16(h, pw_f130);

            // tmp0/1 = (in0 ± in4) << CONST_BITS, computed via the
            // (zero, x) unpack + arithmetic-right-shift trick: an i16
            // value placed in the high 16 bits of an i32 is value * 2^16
            // with sign preserved; shifting right by (16 - CONST_BITS)
            // yields value << CONST_BITS, sign-preserved.
            let in4_0 = _mm256_permute2x128_si256::<0x01>(in0_4, in0_4);
            let pw_1_neg1 = load(PW_1_NEG1.0.as_ptr());
            let in0_neg4 = _mm256_sign_epi16(in0_4, pw_1_neg1);
            let sum_diff = _mm256_add_epi16(in4_0, in0_neg4);
            // (in0+in4) in lo 128, (in0-in4) in hi 128

            let zero = _mm256_setzero_si256();
            let lo = _mm256_unpacklo_epi16(zero, sum_diff);
            let hi = _mm256_unpackhi_epi16(zero, sum_diff);
            // 16 - CONST_BITS = 3
            let tmp0_1_l = _mm256_srai_epi32::<3>(lo);
            let tmp0_1_h = _mm256_srai_epi32::<3>(hi);

            let tmp13_12_l = _mm256_sub_epi32(tmp0_1_l, tmp3_2_l);
            let tmp10_11_l = _mm256_add_epi32(tmp0_1_l, tmp3_2_l);
            let tmp13_12_h = _mm256_sub_epi32(tmp0_1_h, tmp3_2_h);
            let tmp10_11_h = _mm256_add_epi32(tmp0_1_h, tmp3_2_h);

            // -- Odd part --------------------------------------------------
            // z3 = in7+in3 (lo 128); z4 = in5+in1 (hi 128)
            let z3_4 = _mm256_add_epi16(in7_5, in3_1);
            let z4_3 = _mm256_permute2x128_si256::<0x01>(z3_4, z3_4);

            let zl = _mm256_unpacklo_epi16(z3_4, z4_3);
            let zh = _mm256_unpackhi_epi16(z3_4, z4_3);
            let pw_mf078 = load(PW_MF078_F117_F078_F117.0.as_ptr());
            // After madd: lo 128 = z3 = z5 - z3_in * F_1_961
            //             hi 128 = z4 = z5 - z4_in * F_0_390
            let z3_4_l = _mm256_madd_epi16(zl, pw_mf078);
            let z3_4_h = _mm256_madd_epi16(zh, pw_mf078);

            // in71_53 = interleave in7_5 with swapped in3_1 (= in1_3)
            let in1_3 = _mm256_permute2x128_si256::<0x01>(in3_1, in3_1);
            let in71_53_l = _mm256_unpacklo_epi16(in7_5, in1_3);
            let in71_53_h = _mm256_unpackhi_epi16(in7_5, in1_3);

            // tmp0/1 partial:
            //   lo 128 from (in7, in1) × (F_0_298-F_0_899, -F_0_899)
            //   hi 128 from (in5, in3) × (F_2_053-F_2_562, -F_2_562)
            let pw_mf060 = load(PW_MF060_MF089_MF050_MF256.0.as_ptr());
            let part_l = _mm256_madd_epi16(in71_53_l, pw_mf060);
            let part_h = _mm256_madd_epi16(in71_53_h, pw_mf060);
            let tmp0_1_lo = _mm256_add_epi32(part_l, z3_4_l);
            let tmp0_1_hi = _mm256_add_epi32(part_h, z3_4_h);

            // tmp3/2 partial:
            //   lo 128 from (in7, in1) × (-F_0_899, F_1_501-F_0_899)
            //   hi 128 from (in5, in3) × (-F_2_562, F_3_072-F_2_562)
            let pw_mf089 = load(PW_MF089_F060_MF256_F050.0.as_ptr());
            let part_l = _mm256_madd_epi16(in71_53_l, pw_mf089);
            let part_h = _mm256_madd_epi16(in71_53_h, pw_mf089);
            // z4_3 swap of z3_4 (so we add z4 to tmp3, z3 to tmp2)
            let z4_3_l = _mm256_permute2x128_si256::<0x01>(z3_4_l, z3_4_l);
            let z4_3_h = _mm256_permute2x128_si256::<0x01>(z3_4_h, z3_4_h);
            let tmp3_2_lo = _mm256_add_epi32(part_l, z4_3_l);
            let tmp3_2_hi = _mm256_add_epi32(part_h, z4_3_h);

            // -- Final output stage: descale + saturating i32→i16 pack
            // (bias, shift) selected by PASS at codegen time.
            let bias = if PASS == 1 {
                _mm256_loadu_si256(PD_DESCALE_P1.0.as_ptr() as *const __m256i)
            } else {
                _mm256_loadu_si256(IDCT_PD_DESCALE_P2.0.as_ptr() as *const __m256i)
            };

            // data0_1 = pack((tmp10_11 + tmp3_2) descaled)
            let a = _mm256_add_epi32(tmp10_11_l, tmp3_2_lo);
            let b = _mm256_add_epi32(tmp10_11_h, tmp3_2_hi);
            let a = _mm256_add_epi32(a, bias);
            let b = _mm256_add_epi32(b, bias);
            let (a, b) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(a), _mm256_srai_epi32::<11>(b))
            } else {
                (_mm256_srai_epi32::<18>(a), _mm256_srai_epi32::<18>(b))
            };
            let data0_1 = _mm256_packs_epi32(a, b);

            // data7_6 = pack((tmp10_11 - tmp3_2) descaled)
            let a = _mm256_sub_epi32(tmp10_11_l, tmp3_2_lo);
            let b = _mm256_sub_epi32(tmp10_11_h, tmp3_2_hi);
            let a = _mm256_add_epi32(a, bias);
            let b = _mm256_add_epi32(b, bias);
            let (a, b) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(a), _mm256_srai_epi32::<11>(b))
            } else {
                (_mm256_srai_epi32::<18>(a), _mm256_srai_epi32::<18>(b))
            };
            let data7_6 = _mm256_packs_epi32(a, b);

            // data3_2 = pack((tmp13_12 + tmp0_1) descaled)
            let a = _mm256_add_epi32(tmp13_12_l, tmp0_1_lo);
            let b = _mm256_add_epi32(tmp13_12_h, tmp0_1_hi);
            let a = _mm256_add_epi32(a, bias);
            let b = _mm256_add_epi32(b, bias);
            let (a, b) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(a), _mm256_srai_epi32::<11>(b))
            } else {
                (_mm256_srai_epi32::<18>(a), _mm256_srai_epi32::<18>(b))
            };
            let data3_2 = _mm256_packs_epi32(a, b);

            // data4_5 = pack((tmp13_12 - tmp0_1) descaled)
            let a = _mm256_sub_epi32(tmp13_12_l, tmp0_1_lo);
            let b = _mm256_sub_epi32(tmp13_12_h, tmp0_1_hi);
            let a = _mm256_add_epi32(a, bias);
            let b = _mm256_add_epi32(b, bias);
            let (a, b) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(a), _mm256_srai_epi32::<11>(b))
            } else {
                (_mm256_srai_epi32::<18>(a), _mm256_srai_epi32::<18>(b))
            };
            let data4_5 = _mm256_packs_epi32(a, b);

            (data0_1, data3_2, data4_5, data7_6)
        }
    }

    /// IDCT pass specialized to inputs whose "rows" 4..7 are all zero.
    /// Const-generic on pass id (1 or 2) selects the descale shift and
    /// matching round bias, matching `idct_dodct`.
    ///
    /// For PASS = 1 the precondition is that the column-direction
    /// inputs' rows 4..7 are zero (= original block rows 4..7 are
    /// zero). For PASS = 2 the same packing comes from `idct_dotranspose`
    /// applied to a pass-1 output whose columns 4..7 are zero — which
    /// happens whenever the original block columns 4..7 are zero, since
    /// pass-1 transforms each column in isolation.
    ///
    /// Packing matches `idct_dodct`:
    ///   `in0_4`: lo = "row 0", hi = 0     ("row 4" zero)
    ///   `in3_1`: lo = "row 3", hi = "row 1"
    ///   `in2_6`: lo = "row 2", hi = 0     ("row 6" zero)
    ///   `in7_5`: all zero                  ("rows 7, 5" zero)
    /// Returns `(data0_1, data3_2, data4_5, data7_6)` identical to what
    /// `idct_dodct::<PASS>` would return on the same inputs. Mirrors
    /// `idct_pass1_sparse` / `idct_pass2_sparse` in `arch::neon`.
    ///
    /// # Safety
    /// Caller must have AVX2 enabled and must hold the sparse
    /// pre-condition above; passing non-zero "rows" 4..7 produces
    /// incorrect output.
    #[inline(always)]
    unsafe fn idct_dodct_sparse<const PASS: i32>(
        in0_4: __m256i,
        in3_1: __m256i,
        in2_6: __m256i,
        in7_5: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            // -- Even part: tmp3_2 from cols 2, 6 (row 6 = 0) ----------
            // Same madd structure as the regular kernel; the in6_2 swap
            // still carries one valid copy of row 2 into the upper lane
            // (the lower lane of the swap is zero), and the pw_f130
            // constants select tmp3 vs tmp2 by pair element. The
            // by-zero multiplies fall out as no-ops at the i32 sum.
            let in6_2 = _mm256_permute2x128_si256::<0x01>(in2_6, in2_6);
            let l = _mm256_unpacklo_epi16(in2_6, in6_2);
            let h = _mm256_unpackhi_epi16(in2_6, in6_2);
            let pw_f130 = load(PW_F130_F054_MF130_F054.0.as_ptr());
            let tmp3_2_l = _mm256_madd_epi16(l, pw_f130);
            let tmp3_2_h = _mm256_madd_epi16(h, pw_f130);

            // -- tmp0/1 = row 0 << CONST_BITS  (row 4 = 0) -------------
            // In the regular kernel, sum_diff packs (in0+in4, in0-in4)
            // across the 128-bit lanes; with in4 = 0 that collapses to
            // (row0, row0), so we just broadcast row 0 into both halves
            // and run the same (zero, x) widen + arithmetic-shift trick
            // (16 - CONST_BITS = 3) to get value << CONST_BITS with
            // sign preserved.
            let row0_dup = _mm256_permute2x128_si256::<0x00>(in0_4, in0_4);
            let zero = _mm256_setzero_si256();
            let lo = _mm256_unpacklo_epi16(zero, row0_dup);
            let hi = _mm256_unpackhi_epi16(zero, row0_dup);
            let tmp0_1_l = _mm256_srai_epi32::<3>(lo);
            let tmp0_1_h = _mm256_srai_epi32::<3>(hi);

            let tmp13_12_l = _mm256_sub_epi32(tmp0_1_l, tmp3_2_l);
            let tmp10_11_l = _mm256_add_epi32(tmp0_1_l, tmp3_2_l);
            let tmp13_12_h = _mm256_sub_epi32(tmp0_1_h, tmp3_2_h);
            let tmp10_11_h = _mm256_add_epi32(tmp0_1_h, tmp3_2_h);

            // -- Odd part: in7_5 = 0 -----------------------------------
            // z3_4 = in7_5 + in3_1 collapses to in3_1.
            let z3_4 = in3_1;
            let z4_3 = _mm256_permute2x128_si256::<0x01>(z3_4, z3_4);

            let zl = _mm256_unpacklo_epi16(z3_4, z4_3);
            let zh = _mm256_unpackhi_epi16(z3_4, z4_3);
            let pw_mf078 = load(PW_MF078_F117_F078_F117.0.as_ptr());
            let z3_4_l = _mm256_madd_epi16(zl, pw_mf078);
            let z3_4_h = _mm256_madd_epi16(zh, pw_mf078);

            // in71_53 mixes in7_5 (zero) with in1_3; the resulting
            // half-zero pattern still feeds the same constant-pair madd
            // — the zero halves drop out at the i32 sum stage.
            let in1_3 = _mm256_permute2x128_si256::<0x01>(in3_1, in3_1);
            let in71_53_l = _mm256_unpacklo_epi16(in7_5, in1_3);
            let in71_53_h = _mm256_unpackhi_epi16(in7_5, in1_3);

            let pw_mf060 = load(PW_MF060_MF089_MF050_MF256.0.as_ptr());
            let part_l = _mm256_madd_epi16(in71_53_l, pw_mf060);
            let part_h = _mm256_madd_epi16(in71_53_h, pw_mf060);
            let tmp0_1_lo = _mm256_add_epi32(part_l, z3_4_l);
            let tmp0_1_hi = _mm256_add_epi32(part_h, z3_4_h);

            let pw_mf089 = load(PW_MF089_F060_MF256_F050.0.as_ptr());
            let part_l = _mm256_madd_epi16(in71_53_l, pw_mf089);
            let part_h = _mm256_madd_epi16(in71_53_h, pw_mf089);
            let z4_3_l = _mm256_permute2x128_si256::<0x01>(z3_4_l, z3_4_l);
            let z4_3_h = _mm256_permute2x128_si256::<0x01>(z3_4_h, z3_4_h);
            let tmp3_2_lo = _mm256_add_epi32(part_l, z4_3_l);
            let tmp3_2_hi = _mm256_add_epi32(part_h, z4_3_h);

            // -- Final output stage: descale + saturating i32→i16 pack.
            // (bias, shift) selected by PASS at codegen time, matching
            // the regular kernel.
            let bias = if PASS == 1 {
                _mm256_loadu_si256(PD_DESCALE_P1.0.as_ptr() as *const __m256i)
            } else {
                _mm256_loadu_si256(IDCT_PD_DESCALE_P2.0.as_ptr() as *const __m256i)
            };

            let a = _mm256_add_epi32(tmp10_11_l, tmp3_2_lo);
            let b = _mm256_add_epi32(tmp10_11_h, tmp3_2_hi);
            let a = _mm256_add_epi32(a, bias);
            let b = _mm256_add_epi32(b, bias);
            let (a, b) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(a), _mm256_srai_epi32::<11>(b))
            } else {
                (_mm256_srai_epi32::<18>(a), _mm256_srai_epi32::<18>(b))
            };
            let data0_1 = _mm256_packs_epi32(a, b);

            let a = _mm256_sub_epi32(tmp10_11_l, tmp3_2_lo);
            let b = _mm256_sub_epi32(tmp10_11_h, tmp3_2_hi);
            let a = _mm256_add_epi32(a, bias);
            let b = _mm256_add_epi32(b, bias);
            let (a, b) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(a), _mm256_srai_epi32::<11>(b))
            } else {
                (_mm256_srai_epi32::<18>(a), _mm256_srai_epi32::<18>(b))
            };
            let data7_6 = _mm256_packs_epi32(a, b);

            let a = _mm256_add_epi32(tmp13_12_l, tmp0_1_lo);
            let b = _mm256_add_epi32(tmp13_12_h, tmp0_1_hi);
            let a = _mm256_add_epi32(a, bias);
            let b = _mm256_add_epi32(b, bias);
            let (a, b) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(a), _mm256_srai_epi32::<11>(b))
            } else {
                (_mm256_srai_epi32::<18>(a), _mm256_srai_epi32::<18>(b))
            };
            let data3_2 = _mm256_packs_epi32(a, b);

            let a = _mm256_sub_epi32(tmp13_12_l, tmp0_1_lo);
            let b = _mm256_sub_epi32(tmp13_12_h, tmp0_1_hi);
            let a = _mm256_add_epi32(a, bias);
            let b = _mm256_add_epi32(b, bias);
            let (a, b) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(a), _mm256_srai_epi32::<11>(b))
            } else {
                (_mm256_srai_epi32::<18>(a), _mm256_srai_epi32::<18>(b))
            };
            let data4_5 = _mm256_packs_epi32(a, b);

            (data0_1, data3_2, data4_5, data7_6)
        }
    }

    /// AVX2 implementation of `idct_islow`. Bit-exact equivalent of the
    /// scalar reference, modulo the asm-style late saturation that
    /// composes safely with the +128 level-shift for any input.
    ///
    /// # Safety
    /// AVX2 must be available (the runtime gate in `idct_islow`
    /// checks). Inputs are fixed-size references.
    #[target_feature(enable = "avx2")]
    unsafe fn idct_avx2(coef: &[i16; 64], output: &mut [u8; 64]) {
        unsafe {
            let p = coef.as_ptr() as *const __m256i;
            // 4 ymm, each holding 2 contiguous rows of the dequantized
            // coefficient block.
            let in0_1 = _mm256_loadu_si256(p);
            let in2_3 = _mm256_loadu_si256(p.add(1));
            let in4_5 = _mm256_loadu_si256(p.add(2));
            let in6_7 = _mm256_loadu_si256(p.add(3));

            // -----------------------------------------------------------
            // Sparse detection.
            //
            // Row-major OR-reduction over the 4 input ymm registers, then
            // pick out the bits the dispatch needs. Mirrors the NEON
            // detection in `src/arch/neon.rs::idct_islow_inner`, adapted
            // to AVX2's row-pair packing (each ymm = 2 contiguous rows).
            //
            //   rows_4567_zero — rows 4..7 are all zero. Pass-1 columns
            //     only see frequencies 0..3, so a sparse pass-1 kernel
            //     can skip the upper butterflies.
            //   dc_only        — rows 1..7 are all zero AND row 0's AC
            //     lanes (cols 1..7) are zero. Output collapses to a
            //     constant `clamp((dc + 4) >> 3 + 128, 0, 255)` byte and
            //     we skip both passes entirely.
            // -----------------------------------------------------------
            let r4567_or = _mm256_or_si256(in4_5, in6_7);
            let rows_4567_zero = _mm256_testz_si256(r4567_or, r4567_or) != 0;

            // rows 2,3 sit in `in2_3`. Row 1 is the high 128 of `in0_1`;
            // splat it into both halves so the testz covers all 16 i16.
            let row1_dup = _mm256_permute2x128_si256::<0x11>(in0_1, in0_1);
            let r123_or = _mm256_or_si256(row1_dup, in2_3);
            let rows_123_zero = _mm256_testz_si256(r123_or, r123_or) != 0;

            // Row 0 (lo 128 of in0_1), AC lanes = i16 lanes 1..7.
            // Mask lane 0 with -1 (DC) and lanes 1..7 with 0, then
            // ANDNOT to keep only AC.
            let row0 = _mm256_castsi256_si128(in0_1);
            let mask_dc = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, -1);
            let row0_ac = _mm_andnot_si128(mask_dc, row0);
            let row0_ac_zero = _mm_testz_si128(row0_ac, row0_ac) != 0;

            let dc_only = rows_4567_zero && rows_123_zero && row0_ac_zero;

            // -----------------------------------------------------------
            // DC-only fast path: every output sample is the same constant.
            // -----------------------------------------------------------
            if dc_only {
                let dc = coef[0] as i32;
                let val = ((((dc + 4) >> 3) + 128).clamp(0, 255)) as u8;
                let dup = _mm256_set1_epi8(val as i8);
                let outp = output.as_mut_ptr() as *mut __m256i;
                _mm256_storeu_si256(outp, dup);
                _mm256_storeu_si256(outp.add(1), dup);
                _mm256_zeroupper();
                return;
            }

            // Repack so each ymm pairs the rows the DODCT macro wants:
            //   (R0, R4), (R3, R1), (R2, R6), (R7, R5)
            let in0_4 = _mm256_permute2x128_si256::<0x20>(in0_1, in4_5);
            let in3_1 = _mm256_permute2x128_si256::<0x31>(in2_3, in0_1);
            let in2_6 = _mm256_permute2x128_si256::<0x20>(in2_3, in6_7);
            let in7_5 = _mm256_permute2x128_si256::<0x31>(in6_7, in4_5);

            // Pass 1 (columns).
            //
            // Dispatch on the sparse flag: when rows 4..7 are zero,
            // a reduced pass-1 kernel skips the column butterflies that
            // would multiply by zero. Bit-exact with the regular kernel
            // under that pre-condition.
            let (m1, m2, m3, m4) = if rows_4567_zero {
                idct_dodct_sparse::<1>(in0_4, in3_1, in2_6, in7_5)
            } else {
                idct_dodct::<1>(in0_4, in3_1, in2_6, in7_5)
            };
            //   m1 = data0_1, m2 = data3_2, m3 = data4_5, m4 = data7_6
            let (t0, t1, t2, t3) = idct_dotranspose(m1, m2, m3, m4);
            //   t0 = data0_4, t1 = data1_5, t2 = data2_6, t3 = data3_7

            // Between-passes repack: pass 2 wants (in0_4, in3_1, in2_6, in7_5)
            // with the new "rows", which are the transposed pass-1 columns.
            //   in0_4 = t0   (data0_4)
            //   in2_6 = t2   (data2_6)
            //   in3_1 from t3 (data3_7) + t1 (data1_5): pick (data3_lo, data1_lo)
            //   in7_5 from t3 + t1: pick (data7_hi, data5_hi)
            let in7_5_p2 = _mm256_permute2x128_si256::<0x31>(t3, t1);
            let in3_1_p2 = _mm256_permute2x128_si256::<0x20>(t3, t1);

            // Pass 2 (rows). Output values are i16 in [-128, 127]-ish; the
            // saturating pack + 0x80 byte-add below handles the level shift.
            //
            // Sparse dispatch: when the original block's columns 4..7
            // are zero, pass-1 leaves the workspace columns 4..7 zero
            // (each column transforms in isolation), and after the
            // pass-1 transpose those zero columns land in the high
            // 128-bit lane of every `tN`. OR-reduce the four hi lanes
            // and test for zero. `_mm256_permute2x128_si256::<0x11>`
            // pulls the hi 128 of each operand into both 128-bit
            // halves; the testz then covers all eight i16 lanes that
            // matter without a separate extract.
            // Sparse dispatch: when the original block's columns 4..7
            // are zero, pass-1 leaves the workspace columns 4..7 zero
            // (each column transforms in isolation), and after the
            // pass-1 transpose those zero columns land in the high
            // 128-bit lane of every `tN`. OR-reduce the four hi lanes
            // and test for zero. `_mm256_permute2x128_si256::<0x31>`
            // pulls a[1] into the dest low and b[1] into the dest high,
            // so a single permute carries both hi 128 lanes for the
            // testz that follows.
            let t01_hi = _mm256_permute2x128_si256::<0x31>(t0, t1);
            let t23_hi = _mm256_permute2x128_si256::<0x31>(t2, t3);
            let hi_or = _mm256_or_si256(t01_hi, t23_hi);
            let cols_4567_zero_after_p1 = _mm256_testz_si256(hi_or, hi_or) != 0;
            let (m1, m2, m3, m4) = if cols_4567_zero_after_p1 {
                idct_dodct_sparse::<2>(t0, in3_1_p2, t2, in7_5_p2)
            } else {
                idct_dodct::<2>(t0, in3_1_p2, t2, in7_5_p2)
            };
            let (t0, t1, t2, t3) = idct_dotranspose(m1, m2, m3, m4);
            //   t0 = data0_4, t1 = data1_5, t2 = data2_6, t3 = data3_7

            // i16 → i8 saturating pack, two ymm at a time:
            //   pack01_45 lo128 = row0|row1,  hi128 = row4|row5
            //   pack23_67 lo128 = row2|row3,  hi128 = row6|row7
            let pack01_45 = _mm256_packs_epi16(t0, t1);
            let pack23_67 = _mm256_packs_epi16(t2, t3);

            // +128 level shift via wrap-around byte add. After this the
            // i8 lanes are reinterpreted as u8 in [0, 255], matching the
            // scalar reference's clamp(0, 255).
            let center = _mm256_loadu_si256(PB_CENTERJSAMP.0.as_ptr() as *const __m256i);
            let pack01_45 = _mm256_add_epi8(pack01_45, center);
            let pack23_67 = _mm256_add_epi8(pack23_67, center);

            // Reorder so output is contiguous row 0..7:
            //   first  = (row0|row1|row2|row3)
            //   second = (row4|row5|row6|row7)
            let first = _mm256_permute2x128_si256::<0x20>(pack01_45, pack23_67);
            let second = _mm256_permute2x128_si256::<0x31>(pack01_45, pack23_67);

            let outp = output.as_mut_ptr() as *mut __m256i;
            _mm256_storeu_si256(outp, first);
            _mm256_storeu_si256(outp.add(1), second);

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
    use crate::color::PixelLayout;

    #[test]
    fn quant_avx2_matches_scalar_random() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            // No AVX2 on this CPU — runtime dispatch will use scalar
            // anyway, so there is nothing to cross-check.
            return;
        }

        use crate::tables::build_divisors;
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

    // -----------------------------------------------------------------
    // IDCT cross-checks: every byte of the AVX2 output must match the
    // scalar reference for the full input set we care about (DC-only,
    // single AC impulses, ramp, random natural-image-shaped blocks,
    // and saturation extremes).
    // -----------------------------------------------------------------

    fn run_idct_cross(coef: &[i16; 64]) {
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        scalar::dct::idct_islow(coef, &mut a);
        dct::idct_islow(coef, &mut b);
        assert_eq!(a, b, "AVX2 IDCT diverges from scalar for input {coef:?}");
    }

    #[test]
    fn idct_avx2_matches_scalar_zeros() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        run_idct_cross(&[0i16; 64]);
    }

    #[test]
    fn idct_avx2_matches_scalar_dc_only() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Span DC across both signs and a few typical magnitudes.
        for dc in [-2048i16, -256, -1, 0, 1, 256, 2047] {
            let mut block = [0i16; 64];
            block[0] = dc;
            run_idct_cross(&block);
        }
    }

    #[test]
    fn idct_avx2_dc_only_fast_path() {
        // Direct verification of the DC-only fast path: dense DC sweep
        // including saturation extremes and zero. Every input here has
        // all-AC == 0, so the sparse detection must trigger and the
        // fast path must produce a flat block equal to
        // `clamp((dc + 4) >> 3 + 128, 0, 255)`.
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for dc in -2048i16..=2047i16 {
            let mut coef = [0i16; 64];
            coef[0] = dc;
            let expected = ((((dc as i32) + 4) >> 3) + 128).clamp(0, 255) as u8;
            let mut avx_out = [0u8; 64];
            dct::idct_islow(&coef, &mut avx_out);
            for (i, &v) in avx_out.iter().enumerate() {
                assert_eq!(v, expected, "dc={dc} pos={i}");
            }
        }
    }

    #[test]
    fn idct_avx2_sparse_p1_matches_scalar_rows_4567_zero() {
        // Exercises the sparse pass-1 kernel: blocks with rows 4..7
        // all zero. Coefficient magnitudes follow the natural-image
        // shape used by `idct_avx2_matches_scalar_random` (per-row
        // dropoff with frequency) — staying inside the AVX2 kernel's
        // i16-workspace contract — but the high-frequency rows 4..7
        // are forced to zero so the detection flag fires.
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for seed in 0..200u64 {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut coef = [0i16; 64];
            for (k, slot) in coef.iter_mut().enumerate().take(32) {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = (s >> 32) as i32;
                let scale = (1024i32 >> (k / 8)).clamp(1, 2047);
                let span = scale * 2 + 1;
                let centered = r.rem_euclid(span) - scale;
                *slot = centered.clamp(-2047, 2047) as i16;
            }
            run_idct_cross(&coef);
        }
        // Single-coefficient impulses across the kept rows. Magnitudes
        // stay inside the per-row natural range to honour the same
        // i16-workspace contract as the random sweep above.
        for k in 0..32 {
            let scale = (1024i16 >> (k / 8)).max(1);
            for &mag in &[1i16, -1] {
                let mut coef = [0i16; 64];
                coef[k] = mag;
                run_idct_cross(&coef);
            }
            for &mag in &[scale, -scale] {
                let mut coef = [0i16; 64];
                coef[k] = mag;
                run_idct_cross(&coef);
            }
        }
    }

    #[test]
    fn idct_avx2_sparse_p2_matches_scalar_cols_4567_zero_workspace() {
        // Exercises the sparse pass-2 kernel: blocks with cols 4..7
        // all zero, which leaves the workspace columns 4..7 zero after
        // pass-1 (each column transforms in isolation), tripping the
        // post-transpose detection on `t0..t3`. Magnitude distribution
        // mirrors `idct_avx2_sparse_p1_matches_scalar_rows_4567_zero`
        // (per-row dropoff inside the i16-workspace contract).
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for seed in 0..200u64 {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut coef = [0i16; 64];
            for row in 0..8 {
                for col in 0..4 {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let r = (s >> 32) as i32;
                    let scale = (1024i32 >> row).clamp(1, 2047);
                    let span = scale * 2 + 1;
                    let centered = r.rem_euclid(span) - scale;
                    coef[row * 8 + col] = centered.clamp(-2047, 2047) as i16;
                }
            }
            run_idct_cross(&coef);
        }
        // Per-position impulses across the kept columns (cols 0..3) of
        // every row, exercising the sparse pass-2 kernel for each input
        // lane that the precondition leaves nonzero.
        for row in 0..8 {
            for col in 0..4 {
                let k = row * 8 + col;
                let scale = (1024i16 >> row).max(1);
                for &mag in &[1i16, -1, scale, -scale] {
                    let mut coef = [0i16; 64];
                    coef[k] = mag;
                    run_idct_cross(&coef);
                }
            }
        }
        // Combined trigger: rows 4..7 zero AND cols 4..7 zero ⇒ pass-1
        // sparse kernel fires for the column pass, then the pass-2
        // sparse kernel fires for the row pass on the same block.
        for seed in 0..50u64 {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut coef = [0i16; 64];
            for row in 0..4 {
                for col in 0..4 {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let r = (s >> 32) as i32;
                    let scale = (1024i32 >> row).clamp(1, 2047);
                    let span = scale * 2 + 1;
                    let centered = r.rem_euclid(span) - scale;
                    coef[row * 8 + col] = centered.clamp(-2047, 2047) as i16;
                }
            }
            run_idct_cross(&coef);
        }
    }

    #[test]
    fn idct_avx2_matches_scalar_ac_impulses() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Place a unit and a large impulse at every AC position. This
        // exercises every coefficient lane in every column / row pair.
        for k in 1..64 {
            for &mag in &[1i16, -1, 256, -256, 1024, -1024] {
                let mut block = [0i16; 64];
                block[k] = mag;
                run_idct_cross(&block);
            }
        }
    }

    #[test]
    fn idct_avx2_matches_scalar_ramp() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut block = [0i16; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = (i as i16) * 8 - 256;
        }
        run_idct_cross(&block);
    }

    #[test]
    fn idct_avx2_matches_scalar_random() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // 100 natural-image-shaped blocks (dequantized coefficient
        // magnitudes that drop off with frequency). Centered on zero
        // and bounded to ±2047 (i12 raw-DCT spec range) so we stay
        // inside libjpeg-turbo's i16-workspace contract.
        for seed in 0..100u64 {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut block = [0i16; 64];
            for (k, slot) in block.iter_mut().enumerate() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = (s >> 32) as i32;
                let scale = (1024i32 >> (k / 8)).clamp(1, 2047);
                let span = scale * 2 + 1;
                let centered = r.rem_euclid(span) - scale;
                *slot = centered.clamp(-2047, 2047) as i16;
            }
            run_idct_cross(&block);
        }
    }

    #[test]
    fn idct_avx2_matches_scalar_dequantized_real_range() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Simulate real decoder traffic: FDCT a natural-looking 8x8
        // block, quantize via the standard luma table at q=75, then
        // dequantize. The resulting block matches the post-dequant
        // distribution that the decoder feeds into the IDCT. This is
        // the input shape libjpeg-turbo's AVX2 kernel was tuned for —
        // the i16::MAX / all-±2047 saturation panels this test used
        // to cover are out-of-spec for the i16-workspace IDCT and
        // are intentionally outside its contract (see the scalar
        // reference's docstring at `arch::scalar::dct::idct_islow`).
        use crate::tables::build_divisors;
        use crate::tables::{STD_LUMA_QUANT, scale_quant_table};
        let qtab = scale_quant_table(&STD_LUMA_QUANT, 75);
        let div = build_divisors(&qtab);
        for seed in 0..50u64 {
            let mut s = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut src = [0i16; 64];
            for (i, v) in src.iter_mut().enumerate() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((s >> 56) as i16) / 8;
                let grad = ((i / 8) as i16) * 4 + ((i % 8) as i16) * 2 - 64;
                *v = grad + noise;
            }
            scalar::dct::fdct_islow(&mut src);
            let mut quantized = [0i16; 64];
            scalar::quant::quantize_natural(&src, &div, &mut quantized);
            let mut coef = [0i16; 64];
            for i in 0..64 {
                coef[i] = (quantized[i] as i32 * qtab[i] as i32).clamp(-32768, 32767) as i16;
            }
            run_idct_cross(&coef);
        }
    }

    #[test]
    fn h2v2_downsample_avx2_matches_scalar_random() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
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
    fn h2v1_downsample_matches_scalar_random() {
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
    fn h2v2_downsample_avx2_matches_scalar_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // All-zero, all-max, alternating, and a vertical/horizontal
        // gradient panel — exercises the bias rounding on inputs where
        // it matters.
        let panels: [[u8; 256]; 4] = [
            [0u8; 256],
            [255u8; 256],
            {
                let mut a = [0u8; 256];
                for (i, v) in a.iter_mut().enumerate() {
                    *v = if i % 2 == 0 { 1 } else { 0 };
                }
                a
            },
            {
                let mut a = [0u8; 256];
                for (i, v) in a.iter_mut().enumerate() {
                    let row = i / 16;
                    let col = i % 16;
                    *v = ((row * 16 + col) % 256) as u8;
                }
                a
            },
        ];
        for src in &panels {
            let mut a = [0i16; 64];
            let mut b = [0i16; 64];
            scalar::color::h2v2_downsample(src, &mut a);
            color::h2v2_downsample(src, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn color_avx2_matches_scalar_rgba16_random() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::color::RGBA;
        // Deterministic-ish pixel data covering full u8 range.
        let mut pixels = [0u8; 16 * 4];
        for i in 0..16 {
            pixels[i * 4] = ((i * 17) % 256) as u8;
            pixels[i * 4 + 1] = ((i * 23 + 7) % 256) as u8;
            pixels[i * 4 + 2] = ((i * 31 + 13) % 256) as u8;
            pixels[i * 4 + 3] = 255;
        }
        let mut y_s = [0u8; 16];
        let mut cb_s = [0u8; 16];
        let mut cr_s = [0u8; 16];
        scalar::color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_s, &mut cb_s, &mut cr_s);

        let mut y_a = [0u8; 16];
        let mut cb_a = [0u8; 16];
        let mut cr_a = [0u8; 16];
        color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_a, &mut cb_a, &mut cr_a);

        assert_eq!(y_s, y_a);
        assert_eq!(cb_s, cb_a);
        assert_eq!(cr_s, cr_a);
    }

    #[test]
    fn color_avx2_matches_scalar_rgba16_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::color::RGBA;
        let panels: [[u8; 4]; 5] = [
            [0, 0, 0, 255],       // black
            [255, 255, 255, 255], // white
            [255, 0, 0, 255],     // red
            [0, 255, 0, 255],     // green
            [0, 0, 255, 255],     // blue
        ];
        for color in &panels {
            let mut pixels = [0u8; 16 * 4];
            for i in 0..16 {
                pixels[i * 4..i * 4 + 4].copy_from_slice(color);
            }
            let mut y_s = [0u8; 16];
            let mut cb_s = [0u8; 16];
            let mut cr_s = [0u8; 16];
            scalar::color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_s, &mut cb_s, &mut cr_s);
            let mut y_a = [0u8; 16];
            let mut cb_a = [0u8; 16];
            let mut cr_a = [0u8; 16];
            color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_a, &mut cb_a, &mut cr_a);
            assert_eq!(y_s, y_a, "y mismatch for color {color:?}");
            assert_eq!(cb_s, cb_a, "cb mismatch for color {color:?}");
            assert_eq!(cr_s, cr_a, "cr mismatch for color {color:?}");
        }
    }

    #[test]
    fn color_avx2_matches_scalar_rgba16_full_range() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::color::RGBA;
        // Sweep R while keeping G, B varied.
        let mut state: u64 = 0xC0DE;
        for _ in 0..30 {
            let mut pixels = [0u8; 16 * 4];
            for i in 0..16 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                pixels[i * 4] = (state >> 24) as u8;
                pixels[i * 4 + 1] = (state >> 32) as u8;
                pixels[i * 4 + 2] = (state >> 40) as u8;
                pixels[i * 4 + 3] = 255;
            }
            let mut y_s = [0u8; 16];
            let mut cb_s = [0u8; 16];
            let mut cr_s = [0u8; 16];
            scalar::color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_s, &mut cb_s, &mut cr_s);
            let mut y_a = [0u8; 16];
            let mut cb_a = [0u8; 16];
            let mut cr_a = [0u8; 16];
            color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_a, &mut cb_a, &mut cr_a);
            assert_eq!((y_s, cb_s, cr_s), (y_a, cb_a, cr_a));
        }
    }

    #[test]
    fn color_avx2_matches_scalar_rgb24_random() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::color::{BGR, RGB};
        for layout in [RGB, BGR] {
            let mut pixels = [0u8; 16 * 3];
            for i in 0..16 {
                pixels[i * 3] = ((i * 17) % 256) as u8;
                pixels[i * 3 + 1] = ((i * 23 + 7) % 256) as u8;
                pixels[i * 3 + 2] = ((i * 31 + 13) % 256) as u8;
            }
            let mut y_s = [0u8; 16];
            let mut cb_s = [0u8; 16];
            let mut cr_s = [0u8; 16];
            scalar::color::rgb_row_to_ycc(&pixels, layout, 16, &mut y_s, &mut cb_s, &mut cr_s);

            let mut y_a = [0u8; 16];
            let mut cb_a = [0u8; 16];
            let mut cr_a = [0u8; 16];
            color::rgb_row_to_ycc(&pixels, layout, 16, &mut y_a, &mut cb_a, &mut cr_a);

            assert_eq!(y_s, y_a, "y mismatch for {layout:?}");
            assert_eq!(cb_s, cb_a, "cb mismatch for {layout:?}");
            assert_eq!(cr_s, cr_a, "cr mismatch for {layout:?}");
        }
    }

    #[test]
    fn color_avx2_matches_scalar_rgb24_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::color::{BGR, RGB};
        let panels: [[u8; 3]; 5] = [
            [0, 0, 0],       // black
            [255, 255, 255], // white
            [255, 0, 0],     // red
            [0, 255, 0],     // green
            [0, 0, 255],     // blue
        ];
        for layout in [RGB, BGR] {
            for color in &panels {
                let mut pixels = [0u8; 16 * 3];
                for i in 0..16 {
                    pixels[i * 3..i * 3 + 3].copy_from_slice(color);
                }
                let mut y_s = [0u8; 16];
                let mut cb_s = [0u8; 16];
                let mut cr_s = [0u8; 16];
                scalar::color::rgb_row_to_ycc(&pixels, layout, 16, &mut y_s, &mut cb_s, &mut cr_s);
                let mut y_a = [0u8; 16];
                let mut cb_a = [0u8; 16];
                let mut cr_a = [0u8; 16];
                color::rgb_row_to_ycc(&pixels, layout, 16, &mut y_a, &mut cb_a, &mut cr_a);
                assert_eq!(y_s, y_a, "y mismatch for {layout:?} {color:?}");
                assert_eq!(cb_s, cb_a, "cb mismatch for {layout:?} {color:?}");
                assert_eq!(cr_s, cr_a, "cr mismatch for {layout:?} {color:?}");
            }
        }
    }

    #[test]
    fn color_avx2_matches_scalar_rgb24_full_range() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::color::{BGR, RGB};
        for layout in [RGB, BGR] {
            let mut state: u64 = 0xC0DE;
            for _ in 0..30 {
                let mut pixels = [0u8; 16 * 3];
                for i in 0..16 {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    pixels[i * 3] = (state >> 24) as u8;
                    pixels[i * 3 + 1] = (state >> 32) as u8;
                    pixels[i * 3 + 2] = (state >> 40) as u8;
                }
                let mut y_s = [0u8; 16];
                let mut cb_s = [0u8; 16];
                let mut cr_s = [0u8; 16];
                scalar::color::rgb_row_to_ycc(&pixels, layout, 16, &mut y_s, &mut cb_s, &mut cr_s);
                let mut y_a = [0u8; 16];
                let mut cb_a = [0u8; 16];
                let mut cr_a = [0u8; 16];
                color::rgb_row_to_ycc(&pixels, layout, 16, &mut y_a, &mut cb_a, &mut cr_a);
                assert_eq!((y_s, cb_s, cr_s), (y_a, cb_a, cr_a), "{layout:?}");
            }
        }
    }

    #[test]
    fn quant_avx2_matches_scalar_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        use crate::tables::build_divisors;

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

    // ---- ycc_row_to_rgb (decoder color) cross-check ----

    fn lcg_next(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 56) as u8
    }

    fn fill_lcg(buf: &mut [u8], seed: u64) {
        let mut s = seed;
        for v in buf.iter_mut() {
            *v = lcg_next(&mut s);
        }
    }

    /// Run scalar + AVX2 for the same (y, cb, cr) inputs at `layout` and
    /// `n`, then assert byte-identical output.
    fn assert_ycc_match(y: &[u8], cb: &[u8], cr: &[u8], n: usize, layout: PixelLayout, tag: &str) {
        let bpp = layout.bpp;
        let mut out_s = vec![0u8; n * bpp];
        let mut out_a = vec![0u8; n * bpp];
        scalar::color::ycc_row_to_rgb(y, cb, cr, &mut out_s, n, layout);
        color::ycc_row_to_rgb(y, cb, cr, &mut out_a, n, layout);
        assert_eq!(out_s, out_a, "{tag} bpp={bpp}");
    }

    fn all_layouts() -> [(PixelLayout, &'static str); 8] {
        use crate::color::{ABGR, ARGB, BGR, BGRA, BGRX, RGB, RGBA, RGBX};
        [
            (RGB, "RGB"),
            (BGR, "BGR"),
            (RGBA, "RGBA"),
            (BGRA, "BGRA"),
            (ARGB, "ARGB"),
            (ABGR, "ABGR"),
            (RGBX, "RGBX"),
            (BGRX, "BGRX"),
        ]
    }

    #[test]
    fn ycc_row_to_rgb_avx2_matches_scalar_all_layouts_n16() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Random panel that exercises all u8 inputs.
        let mut y = [0u8; 16];
        let mut cb = [0u8; 16];
        let mut cr = [0u8; 16];
        fill_lcg(&mut y, 0xC0DE_0001);
        fill_lcg(&mut cb, 0xC0DE_0002);
        fill_lcg(&mut cr, 0xC0DE_0003);
        for (layout, tag) in all_layouts() {
            assert_ycc_match(&y, &cb, &cr, 16, layout, tag);
        }
    }

    #[test]
    fn ycc_row_to_rgb_avx2_matches_scalar_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Sweep every (y, cb_centered, cr_centered) extreme combination
        // across a 16-pixel block. With 16 lanes we can cover the 8 corner
        // cases of {0, 255}^3 plus 8 mid-range checks in a single block.
        let panels: [(u8, u8, u8); 16] = [
            (0, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
            (255, 0, 255),
            (255, 255, 0),
            (255, 255, 255),
            (128, 128, 128),
            (16, 128, 128),
            (235, 128, 128),
            (128, 16, 128),
            (128, 240, 128),
            (128, 128, 16),
            (128, 128, 240),
            (200, 50, 200),
        ];
        let mut y = [0u8; 16];
        let mut cb = [0u8; 16];
        let mut cr = [0u8; 16];
        for (i, (yi, cbi, cri)) in panels.iter().enumerate() {
            y[i] = *yi;
            cb[i] = *cbi;
            cr[i] = *cri;
        }
        for (layout, tag) in all_layouts() {
            assert_ycc_match(&y, &cb, &cr, 16, layout, tag);
        }
    }

    #[test]
    fn ycc_row_to_rgb_avx2_matches_scalar_chroma_sweep() {
        // Exhaustively cover every (cb, cr) ∈ [0, 256)^2. Per block we
        // pin cb to a constant value and let cr step through a 16-value
        // window, so 256 cb values × 16 cr windows = 4096 blocks covers
        // all 65536 combinations. Y is given LCG jitter to keep the
        // rounding term varied.
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut y = [0u8; 16];
        for cb_val in 0..=255u16 {
            let cb = [cb_val as u8; 16];
            for cr_block in 0..16u16 {
                let mut cr = [0u8; 16];
                for (i, v) in cr.iter_mut().enumerate() {
                    *v = (cr_block * 16 + i as u16) as u8;
                }
                fill_lcg(
                    &mut y,
                    0xF00D_0000 ^ cb_val as u64 ^ ((cr_block as u64) << 8),
                );
                for (layout, tag) in all_layouts() {
                    assert_ycc_match(&y, &cb, &cr, 16, layout, tag);
                }
            }
        }
    }

    #[test]
    fn ycc_row_to_rgb_avx2_matches_scalar_with_tail() {
        // n not a multiple of 16: AVX2 handles the first 16-pixel chunks,
        // the tail (1..15 pixels) falls through to scalar. Verify the
        // boundary by sweeping n through {1, 7, 8, 15, 16, 17, 31, 33, 47}.
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut y = [0u8; 64];
        let mut cb = [0u8; 64];
        let mut cr = [0u8; 64];
        fill_lcg(&mut y, 0xBEEF_0001);
        fill_lcg(&mut cb, 0xBEEF_0002);
        fill_lcg(&mut cr, 0xBEEF_0003);
        for n in [1usize, 7, 8, 15, 16, 17, 31, 32, 33, 47] {
            for (layout, tag) in all_layouts() {
                assert_ycc_match(&y[..n], &cb[..n], &cr[..n], n, layout, tag);
            }
        }
    }

    #[test]
    fn huffman_nonzero_bitmap_sse2_matches_scalar() {
        // SSE2 is x86_64 baseline; no runtime gate needed.

        // All-zero.
        let block = [0i16; 64];
        assert_eq!(
            scalar::huffman::nonzero_bitmap(&block),
            huffman::nonzero_bitmap(&block),
        );

        // All-nonzero.
        let mut block = [0i16; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = (i as i16) - 32;
            if *v == 0 {
                *v = 1;
            }
        }
        assert_eq!(
            scalar::huffman::nonzero_bitmap(&block),
            huffman::nonzero_bitmap(&block),
        );

        // Sparse including 16-lane chunk boundaries.
        let mut block = [0i16; 64];
        for k in [0, 1, 7, 8, 15, 16, 31, 32, 47, 48, 62, 63] {
            block[k] = (k as i16) + 1;
        }
        assert_eq!(
            scalar::huffman::nonzero_bitmap(&block),
            huffman::nonzero_bitmap(&block),
        );

        // i16 extremes.
        let mut block = [0i16; 64];
        block[0] = i16::MIN;
        block[63] = i16::MAX;
        assert_eq!(
            scalar::huffman::nonzero_bitmap(&block),
            huffman::nonzero_bitmap(&block),
        );

        // LCG panel.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let mut block = [0i16; 64];
        for v in block.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((state >> 55) as i16).wrapping_sub(128);
        }
        assert_eq!(
            scalar::huffman::nonzero_bitmap(&block),
            huffman::nonzero_bitmap(&block),
        );
    }

    // -----------------------------------------------------------------
    // sample (fancy chroma upsample) cross-checks
    // -----------------------------------------------------------------

    fn lcg_bytes(seed: u64, n: usize) -> Vec<u8> {
        let mut s = seed;
        let mut v = vec![0u8; n];
        for b in v.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 56) as u8;
        }
        v
    }

    #[test]
    fn h2v2_fancy_vblend_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Cover the head edge (n < 32), exact-chunk boundaries, and
        // misaligned tails to exercise both the AVX2 body and the
        // scalar tail.
        for &n in &[1usize, 7, 16, 31, 32, 33, 47, 64, 65, 96, 127, 200, 257] {
            let cur = lcg_bytes(0xA1B2_C3D4_E5F6_0718, n);
            let nbr = lcg_bytes(0x1234_5678_9ABC_DEF0, n);
            let mut a = vec![0u8; n];
            let mut b = vec![0u8; n];
            scalar::sample::h2v2_fancy_vblend(&cur, &nbr, &mut a, n);
            sample::h2v2_fancy_vblend(&cur, &nbr, &mut b, n);
            assert_eq!(a, b, "n = {n}");
        }
    }

    #[test]
    fn h2v2_fancy_vblend_avx2_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 128;
        let cur = vec![255u8; n];
        let nbr = vec![255u8; n];
        let mut a = vec![0u8; n];
        let mut b = vec![0u8; n];
        scalar::sample::h2v2_fancy_vblend(&cur, &nbr, &mut a, n);
        sample::h2v2_fancy_vblend(&cur, &nbr, &mut b, n);
        assert_eq!(a, b);

        let cur = vec![0u8; n];
        let nbr = vec![255u8; n];
        scalar::sample::h2v2_fancy_vblend(&cur, &nbr, &mut a, n);
        sample::h2v2_fancy_vblend(&cur, &nbr, &mut b, n);
        assert_eq!(a, b);
    }

    #[test]
    fn h2_fancy_upsample_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Cover n below the AVX2 threshold (scalar fallback), the
        // first AVX2-eligible size, exact-chunk multiples, and various
        // tails that touch both the i = n-1 next-clamp and the head
        // prev-clamp.
        for &n in &[
            1usize, 2, 7, 16, 32, 33, 34, 35, 48, 63, 64, 65, 96, 127, 200, 257,
        ] {
            let src = lcg_bytes(0x0F1E_2D3C_4B5A_6978, n);
            let mut a = vec![0u8; 2 * n];
            let mut b = vec![0u8; 2 * n];
            scalar::sample::h2_fancy_upsample(&src, &mut a, n);
            sample::h2_fancy_upsample(&src, &mut b, n);
            assert_eq!(a, b, "n = {n}");
        }
    }

    #[test]
    fn h2_fancy_upsample_avx2_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 96;
        // All-max — verifies saturation does not over-clip.
        let src = vec![255u8; n];
        let mut a = vec![0u8; 2 * n];
        let mut b = vec![0u8; 2 * n];
        scalar::sample::h2_fancy_upsample(&src, &mut a, n);
        sample::h2_fancy_upsample(&src, &mut b, n);
        assert_eq!(a, b, "all-max");

        // Sharp step at the chunk boundary (i = 32) to exercise the
        // prev/next loads across the AVX2 chunk seam.
        let mut src = vec![0u8; n];
        for (k, v) in src.iter_mut().enumerate() {
            *v = if k < 32 { 0 } else { 255 };
        }
        scalar::sample::h2_fancy_upsample(&src, &mut a, n);
        sample::h2_fancy_upsample(&src, &mut b, n);
        assert_eq!(a, b, "step at 32");
    }
}
