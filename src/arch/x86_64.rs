//! x86_64 SIMD kernels — translations of libjpeg-turbo's
//! `simd/x86_64/*-avx2.asm`. See `NOTICE.md`.
//!
//! Backend status:
//!
//! - `quant` — AVX2
//! - `dct` — AVX2
//! - `color::rgb_row_to_ycc` — AVX2 for n=16 RGBA (4:2:0 hot path);
//!   scalar fallback for n=8 / RGB / non-AVX2
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
    /// AVX2 fast path: `n == 16 && layout.bpp == 4` (any of RGBA / BGRA
    /// / ARGB / ABGR / RGBX / BGRX). All other widths and 3-byte input
    /// layouts fall through to the scalar reference.
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
        if n == 16 && layout.bpp == 4 && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                rgba16_avx2(
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

            // Build interleaved-pair ymm registers used by every component.
            let rg_lo = _mm256_unpacklo_epi16(r_u16, g_u16);
            let rg_hi = _mm256_unpackhi_epi16(r_u16, g_u16);
            let bg_lo = _mm256_unpacklo_epi16(b_u16, g_u16);
            let bg_hi = _mm256_unpackhi_epi16(b_u16, g_u16);

            // 0.5 * B and 0.5 * R via the (zero,X) interleave + >>1
            // trick: unpacklo_wd(0, B) places each B[i] in the high 16
            // bits of a u32 lane (= B[i] << 16); then srli 1 gives
            // B[i] << 15 = B[i] * 32768, which is what F_0_500 would do
            // if it fit in i16.
            let zero = _mm256_setzero_si256();
            let half_b_lo = _mm256_srli_epi32::<1>(_mm256_unpacklo_epi16(zero, b_u16));
            let half_b_hi = _mm256_srli_epi32::<1>(_mm256_unpackhi_epi16(zero, b_u16));
            let half_r_lo = _mm256_srli_epi32::<1>(_mm256_unpacklo_epi16(zero, r_u16));
            let half_r_hi = _mm256_srli_epi32::<1>(_mm256_unpackhi_epi16(zero, r_u16));

            // Constants
            let c_y_rg = _mm256_loadu_si256(PW_F0299_F0337.0.as_ptr() as *const __m256i);
            let c_y_bg = _mm256_loadu_si256(PW_F0114_F0250.0.as_ptr() as *const __m256i);
            let c_cb_rg = _mm256_loadu_si256(PW_MF016_MF033.0.as_ptr() as *const __m256i);
            let c_cr_bg = _mm256_loadu_si256(PW_MF008_MF041.0.as_ptr() as *const __m256i);
            let bias_y = _mm256_loadu_si256(PD_ONEHALF.0.as_ptr() as *const __m256i);
            let bias_cbcr = _mm256_loadu_si256(PD_ONEHALFM1_CJ.0.as_ptr() as *const __m256i);

            // Y: madd(R,G; F_0_299, F_0_337) + madd(B,G; F_0_114, F_0_250) + 0.5
            //    The (B, G) madd is the "extra" term.
            let y_extra_lo = _mm256_madd_epi16(bg_lo, c_y_bg);
            let y_extra_hi = _mm256_madd_epi16(bg_hi, c_y_bg);
            let y_u16 = finalize_component(rg_lo, rg_hi, c_y_rg, y_extra_lo, y_extra_hi, bias_y);

            // Cb: madd(R,G; -F_0_168, -F_0_331) + 0.5*B + bias
            let cb_u16 = finalize_component(rg_lo, rg_hi, c_cb_rg, half_b_lo, half_b_hi, bias_cbcr);

            // Cr: madd(B,G; -F_0_081, -F_0_418) + 0.5*R + bias
            let cr_u16 = finalize_component(bg_lo, bg_hi, c_cr_bg, half_r_lo, half_r_hi, bias_cbcr);

            pack_and_store_u16x16(y_u16, y);
            pack_and_store_u16x16(cb_u16, cb);
            pack_and_store_u16x16(cr_u16, cr);

            _mm256_zeroupper();
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
    pub fn ycc_row_to_rgb(
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        out: &mut [u8],
        n: usize,
        layout: PixelLayout,
    ) {
        crate::arch::scalar::color::ycc_row_to_rgb(y, cb, cr, out, n, layout)
    }
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

    /// # Safety
    /// AVX2 must be available (the runtime gate in `quantize_natural`
    /// checks). All inputs are fixed-size references.
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

            // Repack so each ymm pairs the rows the DODCT macro wants:
            //   (R0, R4), (R3, R1), (R2, R6), (R7, R5)
            let in0_4 = _mm256_permute2x128_si256::<0x20>(in0_1, in4_5);
            let in3_1 = _mm256_permute2x128_si256::<0x31>(in2_3, in0_1);
            let in2_6 = _mm256_permute2x128_si256::<0x20>(in2_3, in6_7);
            let in7_5 = _mm256_permute2x128_si256::<0x31>(in6_7, in4_5);

            // Pass 1 (columns).
            let (m1, m2, m3, m4) = idct_dodct::<1>(in0_4, in3_1, in2_6, in7_5);
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
            let (m1, m2, m3, m4) = idct_dodct::<2>(t0, in3_1_p2, t2, in7_5_p2);
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
        // 100 natural-image-shaped blocks (i.e. dequantized coefficient
        // magnitudes that drop off with frequency).
        for seed in 0..100u64 {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut block = [0i16; 64];
            for k in 0..64 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = (s >> 32) as i32;
                let scale = (1024i32 >> (k / 8)).max(1);
                block[k] = (r % (scale * 2 + 1) - scale) as i16;
            }
            run_idct_cross(&block);
        }
    }

    #[test]
    fn idct_avx2_matches_scalar_extremes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // Full i16 saturation pattern — well outside the JPEG-valid
        // dequant range, but the asm's late-saturating pack should still
        // collapse to the same clamped output as scalar.
        let mut block = [0i16; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = if i % 2 == 0 { i16::MAX } else { i16::MIN };
        }
        run_idct_cross(&block);

        // Alternating ±large within typical dequant range too.
        for (i, v) in block.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 4096 } else { -4096 };
        }
        run_idct_cross(&block);
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
}
