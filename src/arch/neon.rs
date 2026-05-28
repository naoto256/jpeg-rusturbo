//! AArch64 NEON kernels — translations of libjpeg-turbo's
//! `simd/arm/jccolor-neon.c`, `jcsample-neon.c`, `jfdctint-neon.c`, and
//! `jquanti-neon.c`. See `NOTICE.md` for the upstream
//! BSD-3-Clause + IJG notice.
//!
//! Output is bit-exact identical to `arch::scalar` — the cross-check
//! tests at the bottom of this file assert that on a panel of inputs.
//!
//! # Safety
//!
//! Every `unsafe { … }` block in this module wraps a `core::arch::aarch64::*`
//! NEON intrinsic. NEON is mandatory on AArch64 per the ARMv8-A architecture
//! reference manual, and this entire module is gated upstream by
//! `#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]` in
//! `src/arch/mod.rs`. The intrinsics therefore have a well-defined CPU-side
//! contract on every reachable target. On the Rust side, each call passes
//! `vld1q_*` / `vst1q_*` pointers derived from `&[i16; 64]` / `&mut [u8]`
//! borrows whose lifetime covers the load/store, with element counts equal
//! to the intrinsic's vector width (16-lane u8 / 8-lane i16 / 4-lane i32).
//! No call reads past the end of a slice and no call writes to an aliased
//! or out-of-bounds destination — the cross-check tests at the bottom of
//! this file would have caught any such bug, since they compare every
//! emitted byte against the scalar reference.

#![allow(dead_code)]

// ===========================================================================
// color: 16-pixel-wide RGB(A) → YCbCr, chroma downsample (4:2:0 and 4:2:2 NEON)
// ===========================================================================
pub mod color {
    use core::arch::aarch64::*;

    use crate::color::{CHROMA_BIAS, PixelLayout};

    /// Constants laid out for `vmull_laneq_u16` indexing 0..=7:
    ///   0: 0.299 R weight (Y)
    ///   1: 0.587 G weight (Y)
    ///   2: 0.114 B weight (Y)
    ///   3: 0.16874 R weight (Cb, subtracted)
    ///   4: 0.33126 G weight (Cb, subtracted)
    ///   5: 0.50000 R/B weight (Cb adds B, Cr adds R)
    ///   6: 0.41869 G weight (Cr, subtracted)
    ///   7: 0.08131 B weight (Cr, subtracted)
    const CONSTS: [u16; 8] = [19595, 38470, 7471, 11059, 21709, 32768, 27439, 5329];

    /// Convert one row of `n` pixels (must be 8 or 16) to YCbCr, chroma
    /// centered on 128. Bit-exact with `arch::scalar::color::rgb_row_to_ycc`.
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
        unsafe {
            if n == 16 {
                rgb_row_16_inner(
                    pixels.as_ptr(),
                    layout,
                    y.as_mut_ptr(),
                    cb.as_mut_ptr(),
                    cr.as_mut_ptr(),
                );
            } else {
                rgb_row_8_inner(
                    pixels.as_ptr(),
                    layout,
                    y.as_mut_ptr(),
                    cb.as_mut_ptr(),
                    cr.as_mut_ptr(),
                );
            }
        }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON is available, so this
    /// function is only "unsafe" in the syntactic `target_feature`
    /// sense. All inputs are by-value vector lanes; no memory access.
    #[target_feature(enable = "neon")]
    unsafe fn rgb16_to_ycc(
        consts: uint16x8_t,
        chroma_bias: uint32x4_t,
        r: uint16x8_t,
        g: uint16x8_t,
        b: uint16x8_t,
    ) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
        // Y = sum * 0.299/0.587/0.114, rounding shr 16.
        let mut y_l = vmull_laneq_u16::<0>(vget_low_u16(r), consts);
        y_l = vmlal_laneq_u16::<1>(y_l, vget_low_u16(g), consts);
        y_l = vmlal_laneq_u16::<2>(y_l, vget_low_u16(b), consts);
        let mut y_h = vmull_laneq_u16::<0>(vget_high_u16(r), consts);
        y_h = vmlal_laneq_u16::<1>(y_h, vget_high_u16(g), consts);
        y_h = vmlal_laneq_u16::<2>(y_h, vget_high_u16(b), consts);

        // Cb = bias - 0.16874 R - 0.33126 G + 0.5 B.
        let mut cb_l = chroma_bias;
        cb_l = vmlsl_laneq_u16::<3>(cb_l, vget_low_u16(r), consts);
        cb_l = vmlsl_laneq_u16::<4>(cb_l, vget_low_u16(g), consts);
        cb_l = vmlal_laneq_u16::<5>(cb_l, vget_low_u16(b), consts);
        let mut cb_h = chroma_bias;
        cb_h = vmlsl_laneq_u16::<3>(cb_h, vget_high_u16(r), consts);
        cb_h = vmlsl_laneq_u16::<4>(cb_h, vget_high_u16(g), consts);
        cb_h = vmlal_laneq_u16::<5>(cb_h, vget_high_u16(b), consts);

        // Cr = bias + 0.5 R - 0.41869 G - 0.08131 B.
        let mut cr_l = chroma_bias;
        cr_l = vmlal_laneq_u16::<5>(cr_l, vget_low_u16(r), consts);
        cr_l = vmlsl_laneq_u16::<6>(cr_l, vget_low_u16(g), consts);
        cr_l = vmlsl_laneq_u16::<7>(cr_l, vget_low_u16(b), consts);
        let mut cr_h = chroma_bias;
        cr_h = vmlal_laneq_u16::<5>(cr_h, vget_high_u16(r), consts);
        cr_h = vmlsl_laneq_u16::<6>(cr_h, vget_high_u16(g), consts);
        cr_h = vmlsl_laneq_u16::<7>(cr_h, vget_high_u16(b), consts);

        // Y uses rounding narrow; Cb/Cr use truncating narrow because
        // the +32767 is already in the bias.
        let y_u16 = vcombine_u16(vrshrn_n_u32::<16>(y_l), vrshrn_n_u32::<16>(y_h));
        let cb_u16 = vcombine_u16(vshrn_n_u32::<16>(cb_l), vshrn_n_u32::<16>(cb_h));
        let cr_u16 = vcombine_u16(vshrn_n_u32::<16>(cr_l), vshrn_n_u32::<16>(cr_h));
        (y_u16, cb_u16, cr_u16)
    }

    /// # Safety
    /// - `inptr` must be readable for at least `16 * layout.bpp` bytes.
    /// - `outy`, `outcb`, `outcr` must each be writable for at least 16 bytes.
    /// - `layout.bpp` must be 3 or 4.
    /// - `target_arch = "aarch64"` guarantees NEON is available.
    #[target_feature(enable = "neon")]
    unsafe fn rgb_row_16_inner(
        inptr: *const u8,
        layout: PixelLayout,
        outy: *mut u8,
        outcb: *mut u8,
        outcr: *mut u8,
    ) {
        unsafe {
            let consts = vld1q_u16(CONSTS.as_ptr());
            let bias = vdupq_n_u32(CHROMA_BIAS);
            // Deinterleave by channel and pick R/G/B by layout offset.
            // `vld3q_u8` / `vld4q_u8` already produce channel-planar
            // vectors; the offset within a pixel byte tuple equals the
            // channel index in the deinterleaved result.
            let (r, g, b) = if layout.bpp == 4 {
                let p = vld4q_u8(inptr);
                let ch = [p.0, p.1, p.2, p.3];
                (ch[layout.r_off], ch[layout.g_off], ch[layout.b_off])
            } else {
                let p = vld3q_u8(inptr);
                let ch = [p.0, p.1, p.2];
                (ch[layout.r_off], ch[layout.g_off], ch[layout.b_off])
            };
            let r_l = vmovl_u8(vget_low_u8(r));
            let g_l = vmovl_u8(vget_low_u8(g));
            let b_l = vmovl_u8(vget_low_u8(b));
            let r_h = vmovl_u8(vget_high_u8(r));
            let g_h = vmovl_u8(vget_high_u8(g));
            let b_h = vmovl_u8(vget_high_u8(b));
            let (y_l, cb_l, cr_l) = rgb16_to_ycc(consts, bias, r_l, g_l, b_l);
            let (y_h, cb_h, cr_h) = rgb16_to_ycc(consts, bias, r_h, g_h, b_h);
            vst1q_u8(outy, vcombine_u8(vmovn_u16(y_l), vmovn_u16(y_h)));
            vst1q_u8(outcb, vcombine_u8(vmovn_u16(cb_l), vmovn_u16(cb_h)));
            vst1q_u8(outcr, vcombine_u8(vmovn_u16(cr_l), vmovn_u16(cr_h)));
        }
    }

    /// # Safety
    /// - `inptr` must be readable for at least `8 * layout.bpp` bytes.
    /// - `outy`, `outcb`, `outcr` must each be writable for at least 8 bytes.
    /// - `layout.bpp` must be 3 or 4.
    /// - `target_arch = "aarch64"` guarantees NEON is available.
    #[target_feature(enable = "neon")]
    unsafe fn rgb_row_8_inner(
        inptr: *const u8,
        layout: PixelLayout,
        outy: *mut u8,
        outcb: *mut u8,
        outcr: *mut u8,
    ) {
        unsafe {
            let consts = vld1q_u16(CONSTS.as_ptr());
            let bias = vdupq_n_u32(CHROMA_BIAS);
            let (r, g, b) = if layout.bpp == 4 {
                let p = vld4_u8(inptr);
                let ch = [p.0, p.1, p.2, p.3];
                (ch[layout.r_off], ch[layout.g_off], ch[layout.b_off])
            } else {
                let p = vld3_u8(inptr);
                let ch = [p.0, p.1, p.2];
                (ch[layout.r_off], ch[layout.g_off], ch[layout.b_off])
            };
            let r = vmovl_u8(r);
            let g = vmovl_u8(g);
            let b = vmovl_u8(b);
            let (y, cb, cr) = rgb16_to_ycc(consts, bias, r, g, b);
            vst1_u8(outy, vmovn_u16(y));
            vst1_u8(outcb, vmovn_u16(cb));
            vst1_u8(outcr, vmovn_u16(cr));
        }
    }

    /// 16x16 → 8x8 box-average chroma downsample with libjpeg's biased
    /// rounding (alternating +1 / +2 across output columns).
    pub fn h2v2_downsample(src: &[u8; 256], dst: &mut [i16; 64]) {
        unsafe { h2v2_inner(src, dst) }
    }

    /// 16x8 → 8x8 horizontal 2:1 chroma downsample with libjpeg's
    /// `{0, 1, 0, 1, ...}` bias. Bit-exact equivalent to
    /// `arch::scalar::color::h2v1_downsample`.
    pub fn h2v1_downsample(src: &[u8; 128], dst: &mut [i16; 64]) {
        unsafe { h2v1_inner(src, dst) }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON is available. `src` /
    /// `dst` are fixed-size references.
    #[target_feature(enable = "neon")]
    unsafe fn h2v1_inner(src: &[u8; 128], dst: &mut [i16; 64]) {
        unsafe {
            // Bias `{0, 1, 0, 1, ...}` over the 8 u16 output lanes per
            // row, packed as u32 lanes (low half 0, high half 1).
            let bias = vreinterpretq_u16_u32(vdupq_n_u32(0x0001_0000));
            let level_shift = vdupq_n_s16(128);
            for j in 0..8 {
                let row = vld1q_u8(src.as_ptr().add(j * 16));
                let sums = vpadalq_u8(bias, row);
                let avg_u8 = vshrn_n_u16::<1>(sums);
                let avg_u16 = vmovl_u8(avg_u8);
                let signed = vsubq_s16(vreinterpretq_s16_u16(avg_u16), level_shift);
                vst1q_s16(dst.as_mut_ptr().add(j * 8), signed);
            }
        }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON is available. `src` /
    /// `dst` are fixed-size references — no caller-side invariants
    /// beyond the standard reference rules.
    #[target_feature(enable = "neon")]
    unsafe fn h2v2_inner(src: &[u8; 256], dst: &mut [i16; 64]) {
        unsafe {
            // Row 0 bias { 1, 0, 1, 0, ... } and row 1 bias { 0, 2, 0, 2, ... }
            // combined into one row pair = { 1, 2, 1, 2, ... } over 16-bit lanes.
            let bias = vreinterpretq_u16_u32(vdupq_n_u32(0x0002_0001));
            let level_shift = vdupq_n_s16(128);
            for j in 0..8 {
                let row0_off = j * 2 * 16;
                let row1_off = row0_off + 16;
                let r0 = vld1q_u8(src.as_ptr().add(row0_off));
                let r1 = vld1q_u8(src.as_ptr().add(row1_off));
                let sums = vpadalq_u8(bias, r0);
                let sums = vpadalq_u8(sums, r1);
                let avg_u8 = vshrn_n_u16::<2>(sums);
                let avg_u16 = vmovl_u8(avg_u8);
                let signed = vsubq_s16(vreinterpretq_s16_u16(avg_u16), level_shift);
                vst1q_s16(dst.as_mut_ptr().add(j * 8), signed);
            }
        }
    }

    // ---- Decoder-side kernels (currently scalar; NEON ports pending) ----

    /// 8x8 → 16x16 box-upsample (decoder-side counterpart of `h2v2_downsample`).
    pub fn h2v2_upsample(src: &[u8; 64], dst: &mut [u8; 256]) {
        crate::arch::scalar::color::h2v2_upsample(src, dst)
    }

    /// 8x8 → 16x8 box-upsample (decoder-side counterpart of `h2v1_downsample`).
    pub fn h2v1_upsample(src: &[u8; 64], dst: &mut [u8; 128]) {
        crate::arch::scalar::color::h2v1_upsample(src, dst)
    }

    /// Fixed-point constants for the inverse color transform, identical
    /// to the ones in `arch::scalar::color::ycc_row_to_rgb`. They do not
    /// fit in `i16`, so the NEON code lifts intermediates to `s32` lanes
    /// and uses `vmulq_n_s32` / `vmlaq_n_s32`.
    const I_CR_R: i32 = 91881; // 1.40200 << 16
    const I_CB_G: i32 = 22554; // 0.34414 << 16
    const I_CR_G: i32 = 46802; // 0.71414 << 16
    const I_CB_B: i32 = 116130; // 1.77200 << 16
    const ICC_HALF: i32 = 1 << 15;

    /// Per-row YCbCr → RGB(A) converter. Processes 16-pixel chunks via
    /// NEON; any tail (`n % 16`) falls back to the scalar reference.
    /// Bit-exact equivalent to `arch::scalar::color::ycc_row_to_rgb`.
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

        let chunks = n / 16;
        let main = chunks * 16;
        if chunks > 0 {
            unsafe {
                ycc_row_main_inner(
                    y.as_ptr(),
                    cb.as_ptr(),
                    cr.as_ptr(),
                    out.as_mut_ptr(),
                    chunks,
                    layout,
                );
            }
        }
        let tail = n - main;
        if tail > 0 {
            crate::arch::scalar::color::ycc_row_to_rgb(
                &y[main..],
                &cb[main..],
                &cr[main..],
                &mut out[main * layout.bpp..],
                tail,
                layout,
            );
        }
    }

    /// Compute R, G, B for one s32x4 batch (4 pixels). `cb_s32` /
    /// `cr_s32` are already `(input - 128)` lifted to s32 lanes.
    #[target_feature(enable = "neon")]
    unsafe fn batch_rgb(
        y_s32: int32x4_t,
        cb_s32: int32x4_t,
        cr_s32: int32x4_t,
        half: int32x4_t,
    ) -> (int32x4_t, int32x4_t, int32x4_t) {
        let r_diff = vshrq_n_s32::<16>(vaddq_s32(vmulq_n_s32(cr_s32, I_CR_R), half));
        let b_diff = vshrq_n_s32::<16>(vaddq_s32(vmulq_n_s32(cb_s32, I_CB_B), half));
        let g_acc = vmlaq_n_s32(vmulq_n_s32(cb_s32, I_CB_G), cr_s32, I_CR_G);
        let g_diff = vshrq_n_s32::<16>(vaddq_s32(g_acc, half));
        (
            vaddq_s32(y_s32, r_diff),
            vsubq_s32(y_s32, g_diff),
            vaddq_s32(y_s32, b_diff),
        )
    }

    /// Widen an s16x8 vector to two s32x4 vectors (low / high halves).
    #[target_feature(enable = "neon")]
    unsafe fn widen_s16x8(v: int16x8_t) -> (int32x4_t, int32x4_t) {
        (vmovl_s16(vget_low_s16(v)), vmovl_s16(vget_high_s16(v)))
    }

    /// Pack four s32x4 RGB components (16 pixels) into one u8x16 channel,
    /// saturating-clamp to `[0, 255]`. The intermediate s32 lanes lie in
    /// ~`[-180, 435]`, comfortably inside s16 range, so the narrow to
    /// s16 cannot overflow; the final `vqmovun_s16` clamps to `[0, 255]`.
    #[target_feature(enable = "neon")]
    unsafe fn pack_channel_u8(
        a: int32x4_t,
        b: int32x4_t,
        c: int32x4_t,
        d: int32x4_t,
    ) -> uint8x16_t {
        let lo = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
        let hi = vcombine_s16(vmovn_s32(c), vmovn_s32(d));
        vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi))
    }

    /// # Safety
    /// - `y_ptr`, `cb_ptr`, `cr_ptr` must each be readable for `chunks * 16` bytes.
    /// - `out_ptr` must be writable for `chunks * 16 * layout.bpp` bytes.
    /// - `layout.bpp` must be 3 or 4.
    /// - `target_arch = "aarch64"` guarantees NEON is available.
    #[target_feature(enable = "neon")]
    unsafe fn ycc_row_main_inner(
        y_ptr: *const u8,
        cb_ptr: *const u8,
        cr_ptr: *const u8,
        out_ptr: *mut u8,
        chunks: usize,
        layout: PixelLayout,
    ) {
        unsafe {
            let half = vdupq_n_s32(ICC_HALF);
            let level_shift = vdupq_n_s16(128);
            let bpp = layout.bpp;

            for k in 0..chunks {
                let y8 = vld1q_u8(y_ptr.add(k * 16));
                let cb8 = vld1q_u8(cb_ptr.add(k * 16));
                let cr8 = vld1q_u8(cr_ptr.add(k * 16));

                // u8 → s16 (8 lanes each, low / high halves).
                let y_l = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y8)));
                let y_h = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y8)));
                let cb_l = vsubq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cb8))),
                    level_shift,
                );
                let cb_h = vsubq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cb8))),
                    level_shift,
                );
                let cr_l = vsubq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cr8))),
                    level_shift,
                );
                let cr_h = vsubq_s16(
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cr8))),
                    level_shift,
                );

                // Widen each s16x8 to two s32x4 (4 sub-batches × 4 lanes = 16 pixels).
                let (y0, y1) = widen_s16x8(y_l);
                let (y2, y3) = widen_s16x8(y_h);
                let (cb0, cb1) = widen_s16x8(cb_l);
                let (cb2, cb3) = widen_s16x8(cb_h);
                let (cr0, cr1) = widen_s16x8(cr_l);
                let (cr2, cr3) = widen_s16x8(cr_h);

                let (r0, g0, b0) = batch_rgb(y0, cb0, cr0, half);
                let (r1, g1, b1) = batch_rgb(y1, cb1, cr1, half);
                let (r2, g2, b2) = batch_rgb(y2, cb2, cr2, half);
                let (r3, g3, b3) = batch_rgb(y3, cb3, cr3, half);

                let r = pack_channel_u8(r0, r1, r2, r3);
                let g = pack_channel_u8(g0, g1, g2, g3);
                let b = pack_channel_u8(b0, b1, b2, b3);

                let dst = out_ptr.add(k * 16 * bpp);
                if bpp == 3 {
                    // RGB (r_off=0) or BGR (r_off=2).
                    let triple = if layout.r_off == 0 {
                        uint8x16x3_t(r, g, b)
                    } else {
                        uint8x16x3_t(b, g, r)
                    };
                    vst3q_u8(dst, triple);
                } else {
                    let alpha = vdupq_n_u8(0xFF);
                    // Channel order in the stored 4-tuple is determined by
                    // (r_off, g_off, b_off, a_off) where a_off is the
                    // remaining index in {0,1,2,3}.
                    let a_off = 6 - layout.r_off - layout.g_off - layout.b_off;
                    let mut ch = [r, g, b, alpha];
                    // Re-order ch so that ch[i] is the channel for output byte i.
                    let mut out_order = [alpha; 4];
                    out_order[layout.r_off] = ch[0];
                    out_order[layout.g_off] = ch[1];
                    out_order[layout.b_off] = ch[2];
                    out_order[a_off] = ch[3];
                    // `ch` consumed; silence unused-mut.
                    let _ = &mut ch;
                    vst4q_u8(
                        dst,
                        uint8x16x4_t(out_order[0], out_order[1], out_order[2], out_order[3]),
                    );
                }
            }
        }
    }
}

// ===========================================================================
// dct: NEON forward DCT (12-mul integer LL&M)
// ===========================================================================
pub mod dct {
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

    /// Inverse 8x8 DCT — NEON intrinsics port of libjpeg-turbo
    /// `jsimd_idct_islow_neon`. Bit-exact equivalent to
    /// `arch::scalar::dct::idct_islow`.
    pub fn idct_islow(coef: &[i16; 64], output: &mut [u8; 64]) {
        unsafe { idct_islow_inner(coef, output) }
    }

    /// NEON forward DCT, in-place. Bit-exact equivalent to
    /// `arch::scalar::dct::fdct_islow`.
    pub fn fdct_islow(data: &mut [i16; 64]) {
        unsafe { fdct_islow_inner(data) }
    }

    // ---- IDCT (jidctint) constants ----
    //
    // Caller-input contract: this kernel mirrors libjpeg-turbo
    // `jsimd_idct_islow_neon`. Intermediate sums of dequantized
    // coefficients are computed in i16, so the caller must ensure every
    // dequantized coefficient fits in `i16` and that pair-sums of
    // odd-row inputs (`row1+row3`, `row5+row7` etc.) also fit in i16.
    // Conforming JPEG bitstreams satisfy this; pathological inputs
    // beyond libjpeg-turbo's documented range may produce wrap-around
    // output. The scalar reference at `arch::scalar::dct::idct_islow`
    // uses an i32 workspace and is safe for any `i16` input — use it
    // (via `--features force-scalar`) if you need the conservative path.

    // Constants packed as 3 × 4 i16 lanes for `vmull_lane_s16` /
    // `vmlal_lane_s16` / `vmlsl_lane_s16` indexing. Layout matches
    // libjpeg-turbo `jsimd_idct_islow_neon_consts`.
    const I_CONSTS: [i16; 12] = [
        // consts1: lane 0..3
        7373,  // F_0_899
        4433,  // F_0_541
        20995, // F_2_562
        -4927, // F_0_298 - F_0_899
        // consts2
        4926,  // F_1_501 - F_0_899
        -4176, // F_2_053 - F_2_562
        10703, // F_0_541 + F_0_765
        9633,  // F_1_175
        // consts3
        6437,   // F_1_175 - F_0_390
        -10704, // F_0_541 - F_1_847
        4177,   // F_3_072 - F_2_562
        -6436,  // F_1_175 - F_1_961
    ];

    const I_CONST_BITS: i32 = 13;
    const I_PASS1_BITS: i32 = 2;
    const I_DESCALE_P1: i32 = I_CONST_BITS - I_PASS1_BITS;
    // Pass-2 total descale is 18 (CONST_BITS + PASS1_BITS + 3); we
    // split it as `vaddhn` (>>16 truncating) + `vqrshrn_n_s16<2>`
    // (>>2 rounding saturating). This is libjpeg-turbo's pattern.
    const I_DESCALE_P2_LATE: i32 = I_CONST_BITS + I_PASS1_BITS + 3 - 16;

    /// jidctint pass-1 regular, 4x8 half. Reads 8 input rows (4 cols
    /// each), writes 16 i16 to each of `ws1` / `ws2` (rows 0-3 →
    /// ws1 transposed, rows 4-7 → ws2 transposed).
    ///
    /// # Safety
    /// NEON enabled; `ws1`/`ws2` must each be writable for 16 i16.
    #[target_feature(enable = "neon")]
    #[inline]
    #[allow(clippy::too_many_arguments)]
    unsafe fn idct_pass1_regular(
        row0: int16x4_t,
        row1: int16x4_t,
        row2: int16x4_t,
        row3: int16x4_t,
        row4: int16x4_t,
        row5: int16x4_t,
        row6: int16x4_t,
        row7: int16x4_t,
        ws1: *mut i16,
        ws2: *mut i16,
    ) {
        unsafe {
            let consts1 = vld1_s16(I_CONSTS.as_ptr());
            let consts2 = vld1_s16(I_CONSTS.as_ptr().add(4));
            let consts3 = vld1_s16(I_CONSTS.as_ptr().add(8));

            // ---- Even part ----
            let z2 = row2;
            let z3 = row6;
            let mut tmp2 = vmull_lane_s16::<1>(z2, consts1); // z2 * F_0_541
            let mut tmp3 = vmull_lane_s16::<2>(z2, consts2); // z2 * (F_0_541+F_0_765)
            tmp2 = vmlal_lane_s16::<1>(tmp2, z3, consts3); // += z3 * (F_0_541-F_1_847)
            tmp3 = vmlal_lane_s16::<1>(tmp3, z3, consts1); // += z3 * F_0_541

            let z2 = row0;
            let z3 = row4;
            let tmp0 = vshll_n_s16::<I_CONST_BITS>(vadd_s16(z2, z3));
            let tmp1 = vshll_n_s16::<I_CONST_BITS>(vsub_s16(z2, z3));

            let tmp10 = vaddq_s32(tmp0, tmp3);
            let tmp13 = vsubq_s32(tmp0, tmp3);
            let tmp11 = vaddq_s32(tmp1, tmp2);
            let tmp12 = vsubq_s32(tmp1, tmp2);

            // ---- Odd part ----
            let t0 = row7;
            let t1 = row5;
            let t2 = row3;
            let t3 = row1;

            let z3_s16 = vadd_s16(t0, t2);
            let z4_s16 = vadd_s16(t1, t3);

            // Refactored per the libjpeg-turbo comment block:
            //   z3 = z3 * (F_1_175 - F_1_961) + z4 * F_1_175
            //   z4 = z3 * F_1_175 + z4 * (F_1_175 - F_0_390)
            let mut z3 = vmull_lane_s16::<3>(z3_s16, consts3); // z3 * (F_1_175-F_1_961)
            let mut z4 = vmull_lane_s16::<3>(z3_s16, consts2); // z3 * F_1_175
            z3 = vmlal_lane_s16::<3>(z3, z4_s16, consts2); // += z4 * F_1_175
            z4 = vmlal_lane_s16::<0>(z4, z4_s16, consts3); // += z4 * (F_1_175-F_0_390)

            // Refactored: tmpN = tmpN * (FIX_N - mult_pair) - tmp_other * mult_pair
            let mut tmp0 = vmull_lane_s16::<3>(t0, consts1); // t0*(F_0_298-F_0_899)
            let mut tmp1 = vmull_lane_s16::<1>(t1, consts2); // t1*(F_2_053-F_2_562)
            let mut tmp2 = vmull_lane_s16::<2>(t2, consts3); // t2*(F_3_072-F_2_562)
            let mut tmp3 = vmull_lane_s16::<0>(t3, consts2); // t3*(F_1_501-F_0_899)

            tmp0 = vmlsl_lane_s16::<0>(tmp0, t3, consts1); // -= t3 * F_0_899
            tmp1 = vmlsl_lane_s16::<2>(tmp1, t2, consts1); // -= t2 * F_2_562
            tmp2 = vmlsl_lane_s16::<2>(tmp2, t1, consts1); // -= t1 * F_2_562
            tmp3 = vmlsl_lane_s16::<0>(tmp3, t0, consts1); // -= t0 * F_0_899

            tmp0 = vaddq_s32(tmp0, z3);
            tmp1 = vaddq_s32(tmp1, z4);
            tmp2 = vaddq_s32(tmp2, z3);
            tmp3 = vaddq_s32(tmp3, z4);

            // Descale by 11 + narrow to i16. For real JPEG coefficients
            // the results fit easily in i16.
            let rows_0123 = int16x4x4_t(
                vrshrn_n_s32::<I_DESCALE_P1>(vaddq_s32(tmp10, tmp3)),
                vrshrn_n_s32::<I_DESCALE_P1>(vaddq_s32(tmp11, tmp2)),
                vrshrn_n_s32::<I_DESCALE_P1>(vaddq_s32(tmp12, tmp1)),
                vrshrn_n_s32::<I_DESCALE_P1>(vaddq_s32(tmp13, tmp0)),
            );
            let rows_4567 = int16x4x4_t(
                vrshrn_n_s32::<I_DESCALE_P1>(vsubq_s32(tmp13, tmp0)),
                vrshrn_n_s32::<I_DESCALE_P1>(vsubq_s32(tmp12, tmp1)),
                vrshrn_n_s32::<I_DESCALE_P1>(vsubq_s32(tmp11, tmp2)),
                vrshrn_n_s32::<I_DESCALE_P1>(vsubq_s32(tmp10, tmp3)),
            );

            // `vst4_s16` writes the 4 vectors interleaved, which is
            // equivalent to transposing the 4x4 block.
            vst4_s16(ws1, rows_0123);
            vst4_s16(ws2, rows_4567);
        }
    }

    /// jidctint pass-2 regular, 4x8 half. Reads 32 i16 from
    /// `workspace`, writes 4 output rows (8 u8 each) at
    /// `output_base + (buf_offset + r) * 8` for r in 0..4.
    ///
    /// # Safety
    /// NEON enabled; `workspace` must be readable for 32 i16;
    /// `output_base` must be writable for `(buf_offset + 4) * 8` bytes.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn idct_pass2_regular(workspace: *const i16, output_base: *mut u8, buf_offset: usize) {
        unsafe {
            let consts1 = vld1_s16(I_CONSTS.as_ptr());
            let consts2 = vld1_s16(I_CONSTS.as_ptr().add(4));
            let consts3 = vld1_s16(I_CONSTS.as_ptr().add(8));

            // ---- Even part. Workspace is laid out as 8 contiguous 4-i16
            // groups (one per "column-of-original" after the pass-1
            // transpose). Index k reads workspace[k*4 .. k*4+4].
            let z2 = vld1_s16(workspace.add(2 * 4));
            let z3 = vld1_s16(workspace.add(6 * 4));
            let mut tmp2 = vmull_lane_s16::<1>(z2, consts1);
            let mut tmp3 = vmull_lane_s16::<2>(z2, consts2);
            tmp2 = vmlal_lane_s16::<1>(tmp2, z3, consts3);
            tmp3 = vmlal_lane_s16::<1>(tmp3, z3, consts1);

            let z2 = vld1_s16(workspace);
            let z3 = vld1_s16(workspace.add(4 * 4));
            let tmp0 = vshll_n_s16::<I_CONST_BITS>(vadd_s16(z2, z3));
            let tmp1 = vshll_n_s16::<I_CONST_BITS>(vsub_s16(z2, z3));

            let tmp10 = vaddq_s32(tmp0, tmp3);
            let tmp13 = vsubq_s32(tmp0, tmp3);
            let tmp11 = vaddq_s32(tmp1, tmp2);
            let tmp12 = vsubq_s32(tmp1, tmp2);

            // ---- Odd part ----
            let t0 = vld1_s16(workspace.add(7 * 4));
            let t1 = vld1_s16(workspace.add(5 * 4));
            let t2 = vld1_s16(workspace.add(3 * 4));
            let t3 = vld1_s16(workspace.add(4));

            let z3_s16 = vadd_s16(t0, t2);
            let z4_s16 = vadd_s16(t1, t3);

            let mut z3 = vmull_lane_s16::<3>(z3_s16, consts3);
            let mut z4 = vmull_lane_s16::<3>(z3_s16, consts2);
            z3 = vmlal_lane_s16::<3>(z3, z4_s16, consts2);
            z4 = vmlal_lane_s16::<0>(z4, z4_s16, consts3);

            let mut tmp0 = vmull_lane_s16::<3>(t0, consts1);
            let mut tmp1 = vmull_lane_s16::<1>(t1, consts2);
            let mut tmp2 = vmull_lane_s16::<2>(t2, consts3);
            let mut tmp3 = vmull_lane_s16::<0>(t3, consts2);

            tmp0 = vmlsl_lane_s16::<0>(tmp0, t3, consts1);
            tmp1 = vmlsl_lane_s16::<2>(tmp1, t2, consts1);
            tmp2 = vmlsl_lane_s16::<2>(tmp2, t1, consts1);
            tmp3 = vmlsl_lane_s16::<0>(tmp3, t0, consts1);

            tmp0 = vaddq_s32(tmp0, z3);
            tmp1 = vaddq_s32(tmp1, z4);
            tmp2 = vaddq_s32(tmp2, z3);
            tmp3 = vaddq_s32(tmp3, z4);

            // Final descale-and-narrow:
            //   total >> 18 with rounding + signed→u8 saturation + +128.
            // Split as `vaddhn_s32` (high-half add: (a+b)>>16, truncating)
            // and `vqrshrn_n_s16<2>` (rounding + saturating narrow to i8).
            // The level shift +128 is applied as a wrap-around `vadd_u8`
            // after reinterpreting the i8 result as u8 — for a saturated
            // i8 in `[-128, 127]`, +128 mod 256 maps exactly to `[0, 255]`.
            let cols_02_s16 = vcombine_s16(vaddhn_s32(tmp10, tmp3), vaddhn_s32(tmp12, tmp1));
            let cols_13_s16 = vcombine_s16(vaddhn_s32(tmp11, tmp2), vaddhn_s32(tmp13, tmp0));
            let cols_46_s16 = vcombine_s16(vsubhn_s32(tmp13, tmp0), vsubhn_s32(tmp11, tmp2));
            let cols_57_s16 = vcombine_s16(vsubhn_s32(tmp12, tmp1), vsubhn_s32(tmp10, tmp3));

            let cols_02_s8 = vqrshrn_n_s16::<I_DESCALE_P2_LATE>(cols_02_s16);
            let cols_13_s8 = vqrshrn_n_s16::<I_DESCALE_P2_LATE>(cols_13_s16);
            let cols_46_s8 = vqrshrn_n_s16::<I_DESCALE_P2_LATE>(cols_46_s16);
            let cols_57_s8 = vqrshrn_n_s16::<I_DESCALE_P2_LATE>(cols_57_s16);

            let center = vdup_n_u8(128);
            let cols_02_u8 = vadd_u8(vreinterpret_u8_s8(cols_02_s8), center);
            let cols_13_u8 = vadd_u8(vreinterpret_u8_s8(cols_13_s8), center);
            let cols_46_u8 = vadd_u8(vreinterpret_u8_s8(cols_46_s8), center);
            let cols_57_u8 = vadd_u8(vreinterpret_u8_s8(cols_57_s8), center);

            // Final 4x8 transpose-and-store using `vst4_lane_u16`.
            // Zipping adjacent pseudo-columns lets us treat output
            // elements as u16 pairs.
            let cols_01_23 = vzip_u8(cols_02_u8, cols_13_u8);
            let cols_45_67 = vzip_u8(cols_46_u8, cols_57_u8);
            let quad = uint16x4x4_t(
                vreinterpret_u16_u8(cols_01_23.0),
                vreinterpret_u16_u8(cols_01_23.1),
                vreinterpret_u16_u8(cols_45_67.0),
                vreinterpret_u16_u8(cols_45_67.1),
            );

            let row0 = output_base.add((buf_offset) * 8).cast::<u16>();
            let row1 = output_base.add((buf_offset + 1) * 8).cast::<u16>();
            let row2 = output_base.add((buf_offset + 2) * 8).cast::<u16>();
            let row3 = output_base.add((buf_offset + 3) * 8).cast::<u16>();
            vst4_lane_u16::<0>(row0, quad);
            vst4_lane_u16::<1>(row1, quad);
            vst4_lane_u16::<2>(row2, quad);
            vst4_lane_u16::<3>(row3, quad);
        }
    }

    /// jidctint pass-1 sparse, 4x8 half — assumes rows 4-7 are all
    /// zero. Same output contract as `idct_pass1_regular`.
    ///
    /// # Safety
    /// NEON enabled; `ws1`/`ws2` must each be writable for 16 i16.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn idct_pass1_sparse(
        row0: int16x4_t,
        row1: int16x4_t,
        row2: int16x4_t,
        row3: int16x4_t,
        ws1: *mut i16,
        ws2: *mut i16,
    ) {
        unsafe {
            let consts1 = vld1_s16(I_CONSTS.as_ptr());
            let consts2 = vld1_s16(I_CONSTS.as_ptr().add(4));
            let consts3 = vld1_s16(I_CONSTS.as_ptr().add(8));

            // Even part — row6 == 0 so z3 drops out.
            let z2 = row2;
            let tmp2 = vmull_lane_s16::<1>(z2, consts1);
            let tmp3 = vmull_lane_s16::<2>(z2, consts2);

            // row4 == 0 so tmp0 == tmp1 == row0 << CONST_BITS.
            let z2 = row0;
            let tmp0 = vshll_n_s16::<I_CONST_BITS>(z2);
            let tmp1 = tmp0;

            let tmp10 = vaddq_s32(tmp0, tmp3);
            let tmp13 = vsubq_s32(tmp0, tmp3);
            let tmp11 = vaddq_s32(tmp1, tmp2);
            let tmp12 = vsubq_s32(tmp1, tmp2);

            // Odd part — t0 = row7 = 0, t1 = row5 = 0. So:
            //   z3 = t2 (= row3)
            //   z4 = t3 (= row1)
            let t2_s16 = row3;
            let t3_s16 = row1;

            let z3_s16 = t2_s16;
            let z4_s16 = t3_s16;

            let mut z3 = vmull_lane_s16::<3>(z3_s16, consts3);
            z3 = vmlal_lane_s16::<3>(z3, z4_s16, consts2);
            let mut z4 = vmull_lane_s16::<3>(z3_s16, consts2);
            z4 = vmlal_lane_s16::<0>(z4, z4_s16, consts3);

            let tmp0 = vmlsl_lane_s16::<0>(z3, t3_s16, consts1);
            let tmp1 = vmlsl_lane_s16::<2>(z4, t2_s16, consts1);
            let tmp2 = vmlal_lane_s16::<2>(z3, t2_s16, consts3);
            let tmp3 = vmlal_lane_s16::<0>(z4, t3_s16, consts2);

            let rows_0123 = int16x4x4_t(
                vrshrn_n_s32::<I_DESCALE_P1>(vaddq_s32(tmp10, tmp3)),
                vrshrn_n_s32::<I_DESCALE_P1>(vaddq_s32(tmp11, tmp2)),
                vrshrn_n_s32::<I_DESCALE_P1>(vaddq_s32(tmp12, tmp1)),
                vrshrn_n_s32::<I_DESCALE_P1>(vaddq_s32(tmp13, tmp0)),
            );
            let rows_4567 = int16x4x4_t(
                vrshrn_n_s32::<I_DESCALE_P1>(vsubq_s32(tmp13, tmp0)),
                vrshrn_n_s32::<I_DESCALE_P1>(vsubq_s32(tmp12, tmp1)),
                vrshrn_n_s32::<I_DESCALE_P1>(vsubq_s32(tmp11, tmp2)),
                vrshrn_n_s32::<I_DESCALE_P1>(vsubq_s32(tmp10, tmp3)),
            );
            vst4_s16(ws1, rows_0123);
            vst4_s16(ws2, rows_4567);
        }
    }

    /// jidctint pass-2 sparse — assumes workspace "rows" 4-7
    /// (= cols 4-7 of original ws after the pass-1 transpose) are
    /// all zero. Same output contract as `idct_pass2_regular`.
    ///
    /// # Safety
    /// NEON enabled; `workspace` readable for 32 i16 (only the
    /// first 16 are accessed); `output_base` writable for
    /// `(buf_offset + 4) * 8` bytes.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn idct_pass2_sparse(workspace: *const i16, output_base: *mut u8, buf_offset: usize) {
        unsafe {
            let consts1 = vld1_s16(I_CONSTS.as_ptr());
            let consts2 = vld1_s16(I_CONSTS.as_ptr().add(4));
            let consts3 = vld1_s16(I_CONSTS.as_ptr().add(8));

            // Even part — index 6 (col 6 of original ws) is zero.
            let z2 = vld1_s16(workspace.add(2 * 4));
            let tmp2 = vmull_lane_s16::<1>(z2, consts1);
            let tmp3 = vmull_lane_s16::<2>(z2, consts2);

            let z2 = vld1_s16(workspace);
            let tmp0 = vshll_n_s16::<I_CONST_BITS>(z2);
            let tmp1 = tmp0;

            let tmp10 = vaddq_s32(tmp0, tmp3);
            let tmp13 = vsubq_s32(tmp0, tmp3);
            let tmp11 = vaddq_s32(tmp1, tmp2);
            let tmp12 = vsubq_s32(tmp1, tmp2);

            // Odd part — index 7 and 5 are zero.
            let t2_s16 = vld1_s16(workspace.add(3 * 4));
            let t3_s16 = vld1_s16(workspace.add(4));

            let z3_s16 = t2_s16;
            let z4_s16 = t3_s16;

            let mut z3 = vmull_lane_s16::<3>(z3_s16, consts3);
            z3 = vmlal_lane_s16::<3>(z3, z4_s16, consts2);
            let mut z4 = vmull_lane_s16::<3>(z3_s16, consts2);
            z4 = vmlal_lane_s16::<0>(z4, z4_s16, consts3);

            let tmp0 = vmlsl_lane_s16::<0>(z3, t3_s16, consts1);
            let tmp1 = vmlsl_lane_s16::<2>(z4, t2_s16, consts1);
            let tmp2 = vmlal_lane_s16::<2>(z3, t2_s16, consts3);
            let tmp3 = vmlal_lane_s16::<0>(z4, t3_s16, consts2);

            let cols_02_s16 = vcombine_s16(vaddhn_s32(tmp10, tmp3), vaddhn_s32(tmp12, tmp1));
            let cols_13_s16 = vcombine_s16(vaddhn_s32(tmp11, tmp2), vaddhn_s32(tmp13, tmp0));
            let cols_46_s16 = vcombine_s16(vsubhn_s32(tmp13, tmp0), vsubhn_s32(tmp11, tmp2));
            let cols_57_s16 = vcombine_s16(vsubhn_s32(tmp12, tmp1), vsubhn_s32(tmp10, tmp3));

            let cols_02_s8 = vqrshrn_n_s16::<I_DESCALE_P2_LATE>(cols_02_s16);
            let cols_13_s8 = vqrshrn_n_s16::<I_DESCALE_P2_LATE>(cols_13_s16);
            let cols_46_s8 = vqrshrn_n_s16::<I_DESCALE_P2_LATE>(cols_46_s16);
            let cols_57_s8 = vqrshrn_n_s16::<I_DESCALE_P2_LATE>(cols_57_s16);

            let center = vdup_n_u8(128);
            let cols_02_u8 = vadd_u8(vreinterpret_u8_s8(cols_02_s8), center);
            let cols_13_u8 = vadd_u8(vreinterpret_u8_s8(cols_13_s8), center);
            let cols_46_u8 = vadd_u8(vreinterpret_u8_s8(cols_46_s8), center);
            let cols_57_u8 = vadd_u8(vreinterpret_u8_s8(cols_57_s8), center);

            let cols_01_23 = vzip_u8(cols_02_u8, cols_13_u8);
            let cols_45_67 = vzip_u8(cols_46_u8, cols_57_u8);
            let quad = uint16x4x4_t(
                vreinterpret_u16_u8(cols_01_23.0),
                vreinterpret_u16_u8(cols_01_23.1),
                vreinterpret_u16_u8(cols_45_67.0),
                vreinterpret_u16_u8(cols_45_67.1),
            );

            let row0 = output_base.add(buf_offset * 8).cast::<u16>();
            let row1 = output_base.add((buf_offset + 1) * 8).cast::<u16>();
            let row2 = output_base.add((buf_offset + 2) * 8).cast::<u16>();
            let row3 = output_base.add((buf_offset + 3) * 8).cast::<u16>();
            vst4_lane_u16::<0>(row0, quad);
            vst4_lane_u16::<1>(row1, quad);
            vst4_lane_u16::<2>(row2, quad);
            vst4_lane_u16::<3>(row3, quad);
        }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON. `coef` / `output` are
    /// fixed-size references; no caller-side invariants beyond the
    /// standard reference rules and the i16-range contract documented
    /// near `I_CONSTS`.
    #[target_feature(enable = "neon")]
    unsafe fn idct_islow_inner(coef: &[i16; 64], output: &mut [u8; 64]) {
        unsafe {
            let p = coef.as_ptr();

            // Load all 16 i16x4 row halves up-front: they are needed
            // either way (sparse detection + regular pass 1).
            let r0l = vld1_s16(p);
            let r0r = vld1_s16(p.add(4));
            let r1l = vld1_s16(p.add(8));
            let r1r = vld1_s16(p.add(12));
            let r2l = vld1_s16(p.add(16));
            let r2r = vld1_s16(p.add(20));
            let r3l = vld1_s16(p.add(24));
            let r3r = vld1_s16(p.add(28));
            let r4l = vld1_s16(p.add(32));
            let r4r = vld1_s16(p.add(36));
            let r5l = vld1_s16(p.add(40));
            let r5r = vld1_s16(p.add(44));
            let r6l = vld1_s16(p.add(48));
            let r6r = vld1_s16(p.add(52));
            let r7l = vld1_s16(p.add(56));
            let r7r = vld1_s16(p.add(60));

            // Sparse-path detection. We classify each 4x8 half by its
            // nonzero structure and pick the cheapest IDCT variant:
            //
            //   left_4567 / right_4567 — rows 4-7 of the half. Zero
            //     ⇒ pass-1 can use the "sparse" variant (rows 4-7
            //     drop out of the butterfly).
            //   right_full — every coefficient in the right 4x8 half.
            //     Zero ⇒ workspace cols 4-7 stay zero after pass-1,
            //     so pass-2 can use its sparse variant on both halves.
            //   bm (full AC bitmap) — every coefficient except `coef[0]`.
            //     Zero ⇒ DC-only fast path: output is a constant
            //     `clamp((dc + 4) >> 3 + 128, 0, 255)` and we skip
            //     both passes entirely.
            //
            // Reduction is a `vget_lane_u64` on the i16x4 OR-reduction
            // — one scalar test per category.
            let left_4567 = vorr_s16(vorr_s16(r4l, r5l), vorr_s16(r6l, r7l));
            let right_4567 = vorr_s16(vorr_s16(r4r, r5r), vorr_s16(r6r, r7r));
            let left_0123_ac = vorr_s16(
                vorr_s16(vset_lane_s16::<0>(0, r0l), r1l),
                vorr_s16(r2l, r3l),
            );
            let right_0123 = vorr_s16(vorr_s16(r0r, r1r), vorr_s16(r2r, r3r));
            let right_full = vorr_s16(right_0123, right_4567);

            let left_4567_zero = vget_lane_u64::<0>(vreinterpret_u64_s16(left_4567)) == 0;
            let right_4567_zero = vget_lane_u64::<0>(vreinterpret_u64_s16(right_4567)) == 0;
            let right_full_zero = vget_lane_u64::<0>(vreinterpret_u64_s16(right_full)) == 0;

            // DC-only requires: rows 1-7 (both halves) zero AND right
            // half row 0 zero AND left row 0 AC lanes zero. Equivalent:
            // `left_0123_ac | left_4567 | right_full == 0`.
            let dc_only_bm = vorr_s16(left_0123_ac, vorr_s16(left_4567, right_full));
            if vget_lane_u64::<0>(vreinterpret_u64_s16(dc_only_bm)) == 0 {
                let dc = vget_lane_s16::<0>(r0l) as i32;
                let val = (((dc + 4) >> 3) + 128).clamp(0, 255) as u8;
                let dup = vdupq_n_u8(val);
                let q = output.as_mut_ptr();
                vst1q_u8(q, dup);
                vst1q_u8(q.add(16), dup);
                vst1q_u8(q.add(32), dup);
                vst1q_u8(q.add(48), dup);
                return;
            }

            let mut ws_l = [0i16; 32];
            let mut ws_r = [0i16; 32];

            // Pass 1, left 4x8 half — sparse if rows 4-7 zero.
            if left_4567_zero {
                idct_pass1_sparse(r0l, r1l, r2l, r3l, ws_l.as_mut_ptr(), ws_r.as_mut_ptr());
            } else {
                idct_pass1_regular(
                    r0l,
                    r1l,
                    r2l,
                    r3l,
                    r4l,
                    r5l,
                    r6l,
                    r7l,
                    ws_l.as_mut_ptr(),
                    ws_r.as_mut_ptr(),
                );
            }

            // Pass 1, right 4x8 half — sparse if rows 4-7 zero. (If the
            // entire half is zero, the sparse path produces all-zero
            // workspace, which is also what pass-2 sparse expects.)
            if right_4567_zero {
                idct_pass1_sparse(
                    r0r,
                    r1r,
                    r2r,
                    r3r,
                    ws_l.as_mut_ptr().add(16),
                    ws_r.as_mut_ptr().add(16),
                );
            } else {
                idct_pass1_regular(
                    r0r,
                    r1r,
                    r2r,
                    r3r,
                    r4r,
                    r5r,
                    r6r,
                    r7r,
                    ws_l.as_mut_ptr().add(16),
                    ws_r.as_mut_ptr().add(16),
                );
            }

            // Pass 2 — sparse for both halves if the right input half
            // was fully zero (workspace cols 4-7 are then zero, which
            // is what pass-2 sparse requires). The pass-1 transposed
            // layout means pass 2's "row N" inputs are actually
            // column N of the original ws.
            let out = output.as_mut_ptr();
            if right_full_zero {
                idct_pass2_sparse(ws_l.as_ptr(), out, 0);
                idct_pass2_sparse(ws_r.as_ptr(), out, 4);
            } else {
                idct_pass2_regular(ws_l.as_ptr(), out, 0);
                idct_pass2_regular(ws_r.as_ptr(), out, 4);
            }
        }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON. `data` is a
    /// fixed-size mut reference; no caller-side invariants beyond
    /// the standard reference rules.
    #[target_feature(enable = "neon")]
    unsafe fn fdct_islow_inner(data: &mut [i16; 64]) {
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

// ===========================================================================
// sample: decoder-side fancy chroma upsample (NEON kernel)
// ===========================================================================
//
// Ports libjpeg-turbo `simd/arm/jdsample-neon.c`. The vertical blend is a
// straight `(3*cur + nbr + 2) >> 2` on 16-byte chunks; the horizontal pass
// loads cur / prev / next via unaligned reads and stores even / odd output
// pairs through `vst2q_u8`. Edges (src[-1] = src[0], src[n] = src[n-1])
// stay on the scalar path so the hot loop is branch-free.
pub mod sample {
    use core::arch::aarch64::*;

    pub fn h2v2_fancy_vblend(cur: &[u8], nbr: &[u8], out: &mut [u8], n: usize) {
        debug_assert!(cur.len() >= n);
        debug_assert!(nbr.len() >= n);
        debug_assert!(out.len() >= n);
        unsafe { h2v2_fancy_vblend_inner(cur, nbr, out, n) }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON. The debug_asserts in
    /// the public wrapper bound `n` against all three slice lengths.
    #[target_feature(enable = "neon")]
    unsafe fn h2v2_fancy_vblend_inner(cur: &[u8], nbr: &[u8], out: &mut [u8], n: usize) {
        unsafe {
            let cp = cur.as_ptr();
            let np = nbr.as_ptr();
            let op = out.as_mut_ptr();
            let mut i = 0usize;
            while i + 16 <= n {
                let c = vld1q_u8(cp.add(i));
                let nb = vld1q_u8(np.add(i));
                let c_lo = vmovl_u8(vget_low_u8(c));
                let c_hi = vmovl_u8(vget_high_u8(c));
                let n_lo = vmovl_u8(vget_low_u8(nb));
                let n_hi = vmovl_u8(vget_high_u8(nb));
                let t_lo = vaddq_u16(vmulq_n_u16(c_lo, 3), n_lo);
                let t_hi = vaddq_u16(vmulq_n_u16(c_hi, 3), n_hi);
                // vrshrn_n_u16(x, 2) = ((x + 2) >> 2) narrowed to u8.
                let r = vcombine_u8(vrshrn_n_u16::<2>(t_lo), vrshrn_n_u16::<2>(t_hi));
                vst1q_u8(op.add(i), r);
                i += 16;
            }
            while i < n {
                let c = *cp.add(i) as u16;
                let nb = *np.add(i) as u16;
                *op.add(i) = ((c * 3 + nb + 2) >> 2) as u8;
                i += 1;
            }
        }
    }

    pub fn h2_fancy_upsample(src: &[u8], dst: &mut [u8], n: usize) {
        debug_assert!(n >= 1);
        debug_assert!(src.len() >= n);
        debug_assert!(dst.len() >= 2 * n);
        // The vector main loop needs i in [1, n - 17] inclusive, so it
        // only triggers for n >= 18. Smaller inputs fall back to scalar.
        if n < 18 {
            crate::arch::scalar::sample::h2_fancy_upsample(src, dst, n);
            return;
        }
        unsafe { h2_fancy_upsample_inner(src, dst, n) }
    }

    /// # Safety
    /// Caller must ensure `n >= 18`, `src.len() >= n`, `dst.len() >= 2*n`.
    /// `target_arch = "aarch64"` guarantees NEON.
    #[target_feature(enable = "neon")]
    unsafe fn h2_fancy_upsample_inner(src: &[u8], dst: &mut [u8], n: usize) {
        unsafe {
            let sp = src.as_ptr();
            let dp = dst.as_mut_ptr();

            // i = 0 edge: src[-1] is clamped to src[0], so the even
            // output collapses to src[0] = (4*s0 + 2) >> 2.
            let s0 = *sp as u16;
            let s1 = *sp.add(1) as u16;
            *dp = s0 as u8;
            *dp.add(1) = ((s0 * 3 + s1 + 2) >> 2) as u8;

            // Vectorize i in [1, n - 17]. Each iteration reads
            // src[i-1 .. i+16] and writes dst[2i .. 2i+32].
            let last_vec_i = n - 17;
            let mut i = 1usize;
            while i <= last_vec_i {
                let cur = vld1q_u8(sp.add(i));
                let prev = vld1q_u8(sp.add(i - 1));
                let next = vld1q_u8(sp.add(i + 1));
                let cur_lo = vmovl_u8(vget_low_u8(cur));
                let cur_hi = vmovl_u8(vget_high_u8(cur));
                let prev_lo = vmovl_u8(vget_low_u8(prev));
                let prev_hi = vmovl_u8(vget_high_u8(prev));
                let next_lo = vmovl_u8(vget_low_u8(next));
                let next_hi = vmovl_u8(vget_high_u8(next));
                let cur3_lo = vmulq_n_u16(cur_lo, 3);
                let cur3_hi = vmulq_n_u16(cur_hi, 3);
                let even = vcombine_u8(
                    vrshrn_n_u16::<2>(vaddq_u16(cur3_lo, prev_lo)),
                    vrshrn_n_u16::<2>(vaddq_u16(cur3_hi, prev_hi)),
                );
                let odd = vcombine_u8(
                    vrshrn_n_u16::<2>(vaddq_u16(cur3_lo, next_lo)),
                    vrshrn_n_u16::<2>(vaddq_u16(cur3_hi, next_hi)),
                );
                vst2q_u8(dp.add(2 * i), uint8x16x2_t(even, odd));
                i += 16;
            }

            // Scalar tail covers the remainder and the i = n-1 edge
            // (next clamped to cur).
            while i < n {
                let cur = *sp.add(i) as u16;
                let prev = *sp.add(i - 1) as u16;
                let next = if i + 1 >= n {
                    cur
                } else {
                    *sp.add(i + 1) as u16
                };
                *dp.add(2 * i) = ((cur * 3 + prev + 2) >> 2) as u8;
                *dp.add(2 * i + 1) = ((cur * 3 + next + 2) >> 2) as u8;
                i += 1;
            }
        }
    }
}

// ===========================================================================
// quant: NEON quantize, natural-order output
// ===========================================================================
pub mod quant {
    use core::arch::aarch64::*;

    use crate::quant::Divisors;

    /// Quantize a 64-element block using the precomputed divisors, in
    /// natural (DCT) order. Bit-exact equivalent to
    /// `arch::scalar::quant::quantize_natural`.
    pub fn quantize_natural(block: &[i16; 64], div: &Divisors, out: &mut [i16; 64]) {
        unsafe { quantize_inner(block, div, out) }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON. All inputs are
    /// fixed-size references; no caller-side invariants beyond the
    /// standard reference rules.
    #[target_feature(enable = "neon")]
    unsafe fn quantize_inner(workspace: &[i16; 64], div: &Divisors, out: &mut [i16; 64]) {
        unsafe {
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
        }
    }

    // ---- zigzag scatter ----------------------------------------------------
    //
    // 8 × 16-byte output chunks (= 8 × 8 i16 = 64 i16 = whole `zz` buffer).
    // For each chunk: two `vqtbl4q_u8` lookups against the low / high halves
    // of `natural`, OR-merged. Out-of-range TBL4 indices (>= 64) produce 0,
    // which is what makes the OR-merge work: each lo-mask byte that should
    // pull from the high half is set to 0xFF and the corresponding hi-mask
    // byte does the real lookup (and vice versa).
    //
    // The const mask tables come from `ZIGZAG` and are baked at compile time
    // (= 16 × 16 = 256 B of `.rodata`). Loading them inside the loop costs
    // 16 `vld1q_u8`s but those overlap with `vqtbl4q_u8` latency.
    //
    // Replaces the scalar `for k in 0..64 { zz[k] = natural[ZIGZAG[k]] }`
    // loop in `quant::quantize_and_zigzag` (was ~96 dynamic insn → ~32 here).
    const fn build_zigzag_masks() -> ([[u8; 16]; 8], [[u8; 16]; 8]) {
        let mut lo = [[0xFFu8; 16]; 8];
        let mut hi = [[0xFFu8; 16]; 8];
        let mut chunk = 0;
        while chunk < 8 {
            let mut j = 0;
            while j < 8 {
                let src = crate::tables::ZIGZAG[chunk * 8 + j];
                if src < 32 {
                    lo[chunk][j * 2] = (src * 2) as u8;
                    lo[chunk][j * 2 + 1] = (src * 2 + 1) as u8;
                } else {
                    let off = (src - 32) * 2;
                    hi[chunk][j * 2] = off as u8;
                    hi[chunk][j * 2 + 1] = (off + 1) as u8;
                }
                j += 1;
            }
            chunk += 1;
        }
        (lo, hi)
    }

    static ZIGZAG_MASKS: ([[u8; 16]; 8], [[u8; 16]; 8]) = build_zigzag_masks();

    /// Permute `natural` (DCT natural order) into `zz` (zig-zag order).
    /// Bit-exact equivalent to `arch::scalar::quant::zigzag_scatter`.
    pub fn zigzag_scatter(natural: &[i16; 64], zz: &mut [i16; 64]) {
        unsafe { zigzag_scatter_inner(natural, zz) }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON. All inputs are fixed-size
    /// references.
    #[target_feature(enable = "neon")]
    unsafe fn zigzag_scatter_inner(natural: &[i16; 64], zz: &mut [i16; 64]) {
        unsafe {
            let nat = natural.as_ptr() as *const u8;
            let dst = zz.as_mut_ptr() as *mut u8;
            let lo_table = uint8x16x4_t(
                vld1q_u8(nat),
                vld1q_u8(nat.add(16)),
                vld1q_u8(nat.add(32)),
                vld1q_u8(nat.add(48)),
            );
            let hi_table = uint8x16x4_t(
                vld1q_u8(nat.add(64)),
                vld1q_u8(nat.add(80)),
                vld1q_u8(nat.add(96)),
                vld1q_u8(nat.add(112)),
            );
            let (ref lo_masks, ref hi_masks) = ZIGZAG_MASKS;
            for k in 0..8 {
                let lo_m = vld1q_u8(lo_masks[k].as_ptr());
                let hi_m = vld1q_u8(hi_masks[k].as_ptr());
                let chunk = vorrq_u8(vqtbl4q_u8(lo_table, lo_m), vqtbl4q_u8(hi_table, hi_m));
                vst1q_u8(dst.add(k * 16), chunk);
            }
        }
    }
}

// ===========================================================================
// huffman: 64-bit nonzero bitmap for AC scan
// ===========================================================================
pub mod huffman {
    use core::arch::aarch64::*;

    /// Bit `k` is set iff `block[k] != 0`. Builds the bitmap 16 lanes
    /// per iteration: `vceqzq_s16` produces 16-bit all-ones / all-zeros
    /// masks per lane, `vmovn_u16` truncates to 8-bit lanes, AND with a
    /// per-lane bit selector packs each lane into a distinct bit, and a
    /// pair of `vaddv_u8` sum-reductions extracts the 16-bit bitmap byte
    /// pair for that chunk.
    ///
    /// AArch64 has no direct `PMOVMSKB` equivalent; the `vaddv` reduction
    /// is the standard substitute. Four iterations cover the full 64
    /// coefficients.
    pub fn nonzero_bitmap(block: &[i16; 64]) -> u64 {
        unsafe { nonzero_bitmap_inner(block) }
    }

    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON. `block` is a fixed-size
    /// reference; no caller-side invariants beyond the standard
    /// reference rules.
    #[target_feature(enable = "neon")]
    unsafe fn nonzero_bitmap_inner(block: &[i16; 64]) -> u64 {
        unsafe {
            const BIT_SELECT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let bit_select = vld1q_u8(BIT_SELECT.as_ptr());
            let mut bm: u64 = 0;
            for chunk in 0..4 {
                let p = block.as_ptr().add(chunk * 16);
                let v0 = vld1q_s16(p);
                let v1 = vld1q_s16(p.add(8));
                // 0xFFFF if zero, 0x0000 if nonzero — invert for "nonzero".
                let nz0 = vmvnq_u16(vceqzq_s16(v0));
                let nz1 = vmvnq_u16(vceqzq_s16(v1));
                // Narrow each 16-bit lane to 8-bit (low byte). For
                // all-ones / all-zeros input this preserves the mask.
                let nz = vcombine_u8(vmovn_u16(nz0), vmovn_u16(nz1));
                // AND with `{1, 2, 4, ..., 128}` per 8-lane half ⇒ each
                // surviving byte holds a single distinct bit.
                let masked = vandq_u8(nz, bit_select);
                // Sum-reduce per half: yields the 8-bit bitmap byte for
                // that half (since all bits are at distinct positions
                // within a u8, sum == bitwise-OR).
                let lo = vaddv_u8(vget_low_u8(masked));
                let hi = vaddv_u8(vget_high_u8(masked));
                let chunk_bits = (lo as u16) | ((hi as u16) << 8);
                bm |= (chunk_bits as u64) << (chunk * 16);
            }
            bm
        }
    }
}

// ===========================================================================
// Cross-check tests: NEON ↔ scalar bit-exact equivalence on a panel of
// inputs. Compiled only on aarch64 (where both backends are reachable).
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::scalar;
    use crate::color::{ABGR, ARGB, BGR, BGRA, BGRX, PixelLayout, RGB, RGBA, RGBX};

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

    fn ycc_xcheck_layout(layout: PixelLayout, n: usize, seed: u64) {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        let mut y = vec![0u8; n];
        let mut cb = vec![0u8; n];
        let mut cr = vec![0u8; n];
        for i in 0..n {
            y[i] = next();
            cb[i] = next();
            cr[i] = next();
        }
        let mut out_s = vec![0u8; n * layout.bpp];
        let mut out_n = vec![0u8; n * layout.bpp];
        scalar::color::ycc_row_to_rgb(&y, &cb, &cr, &mut out_s, n, layout);
        color::ycc_row_to_rgb(&y, &cb, &cr, &mut out_n, n, layout);
        assert_eq!(out_s, out_n, "layout={:?} n={n} seed={seed}", layout);
    }

    #[test]
    fn ycc_neon_matches_scalar_all_layouts() {
        // Edge values + multi-chunk: tail handling, boundary samples,
        // and the eight pixel layouts.
        for &layout in &[RGB, BGR, RGBA, BGRA, ARGB, ABGR, RGBX, BGRX] {
            for &n in &[1usize, 7, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 100] {
                for seed in 0..3u64 {
                    ycc_xcheck_layout(layout, n, seed);
                }
            }
        }
    }

    #[test]
    fn ycc_neon_matches_scalar_extremes() {
        // Saturation: pure-black, pure-white, and the corners of Cb/Cr.
        for &layout in &[RGB, BGRA, ARGB, RGBX] {
            for n in [16usize, 17, 32, 47] {
                let y = vec![0u8; n];
                let cb = vec![0u8; n];
                let cr = vec![255u8; n];
                let mut a = vec![0u8; n * layout.bpp];
                let mut b = vec![0u8; n * layout.bpp];
                scalar::color::ycc_row_to_rgb(&y, &cb, &cr, &mut a, n, layout);
                color::ycc_row_to_rgb(&y, &cb, &cr, &mut b, n, layout);
                assert_eq!(a, b, "extreme y=0,cb=0,cr=255 layout={:?} n={n}", layout);

                let y = vec![255u8; n];
                let cb = vec![255u8; n];
                let cr = vec![0u8; n];
                scalar::color::ycc_row_to_rgb(&y, &cb, &cr, &mut a, n, layout);
                color::ycc_row_to_rgb(&y, &cb, &cr, &mut b, n, layout);
                assert_eq!(a, b, "extreme y=255,cb=255,cr=0 layout={:?} n={n}", layout);
            }
        }
    }

    #[test]
    #[ignore = "kernel micro-bench, run with --ignored --nocapture"]
    fn ycc_neon_micro_bench() {
        use std::time::Instant;
        let n = 16 * 256;
        let iters = 100_000;
        let y: Vec<u8> = (0..n).map(|i| (i * 37 % 256) as u8).collect();
        let cb: Vec<u8> = (0..n).map(|i| (i * 71 % 256) as u8).collect();
        let cr: Vec<u8> = (0..n).map(|i| (i * 113 % 256) as u8).collect();
        let mut out = vec![0u8; n * 4];
        let t = Instant::now();
        for _ in 0..iters {
            scalar::color::ycc_row_to_rgb(&y, &cb, &cr, &mut out, n, RGBA);
            std::hint::black_box(&out);
        }
        let s = t.elapsed().as_nanos() as f64 / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            color::ycc_row_to_rgb(&y, &cb, &cr, &mut out, n, RGBA);
            std::hint::black_box(&out);
        }
        let nv = t.elapsed().as_nanos() as f64 / iters as f64;
        println!(
            "  ycc_row_to_rgb n={n}: scalar={:.2}us neon={:.2}us speedup={:.2}x",
            s / 1000.0,
            nv / 1000.0,
            s / nv
        );
    }

    #[test]
    fn color_neon_matches_scalar_row16() {
        // Deterministic gradient + alternating pattern.
        let mut pixels = [0u8; 16 * 4];
        for i in 0..16 {
            pixels[i * 4] = (i * 17) as u8;
            pixels[i * 4 + 1] = ((i * 23 + 7) % 256) as u8;
            pixels[i * 4 + 2] = ((i * 31 + 13) % 256) as u8;
            pixels[i * 4 + 3] = 255;
        }
        let mut y_s = [0u8; 16];
        let mut cb_s = [0u8; 16];
        let mut cr_s = [0u8; 16];
        scalar::color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_s, &mut cb_s, &mut cr_s);

        let mut y_n = [0u8; 16];
        let mut cb_n = [0u8; 16];
        let mut cr_n = [0u8; 16];
        color::rgb_row_to_ycc(&pixels, RGBA, 16, &mut y_n, &mut cb_n, &mut cr_n);

        assert_eq!(y_s, y_n);
        assert_eq!(cb_s, cb_n);
        assert_eq!(cr_s, cr_n);
    }

    #[test]
    fn color_neon_matches_scalar_downsample() {
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
    fn color_neon_matches_scalar_h2v1_downsample() {
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
    fn fdct_neon_matches_scalar_zeros() {
        let mut a = [0i16; 64];
        let mut b = [0i16; 64];
        scalar::dct::fdct_islow(&mut a);
        dct::fdct_islow(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn fdct_neon_matches_scalar_const() {
        let mut a = [42i16; 64];
        let mut b = [42i16; 64];
        scalar::dct::fdct_islow(&mut a);
        dct::fdct_islow(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn fdct_neon_matches_scalar_ramp() {
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
    fn fdct_neon_matches_scalar_random() {
        for seed in 0..5u64 {
            let mut a = random_block(seed);
            let mut b = a;
            scalar::dct::fdct_islow(&mut a);
            dct::fdct_islow(&mut b);
            assert_eq!(a, b, "seed={seed}");
        }
    }

    #[test]
    fn fdct_neon_matches_scalar_extremes() {
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
    fn quant_neon_matches_scalar() {
        use crate::quant::build_divisors;
        use crate::tables::{STD_LUMA_QUANT, scale_quant_table};

        let mut block = [0i16; 64];
        for (i, v) in block.iter_mut().enumerate() {
            // Vary sign and magnitude.
            let m = (i as i32 * 37) % 4001 - 2000;
            *v = m as i16;
        }
        let qtab = scale_quant_table(&STD_LUMA_QUANT, 80);
        let div = build_divisors(&qtab);

        let mut sout = [0i16; 64];
        let mut nout = [0i16; 64];
        scalar::quant::quantize_natural(&block, &div, &mut sout);
        quant::quantize_natural(&block, &div, &mut nout);
        assert_eq!(sout, nout);
    }

    #[test]
    fn zigzag_scatter_neon_matches_scalar() {
        // Fill with a non-trivial pattern so a permutation bug shows up
        // anywhere in the 64-entry output, not just at zero positions.
        let mut natural = [0i16; 64];
        for (i, v) in natural.iter_mut().enumerate() {
            let m = (i as i32 * 73) % 9001 - 4500;
            *v = m as i16;
        }
        let mut sout = [0i16; 64];
        let mut nout = [0i16; 64];
        scalar::quant::zigzag_scatter(&natural, &mut sout);
        quant::zigzag_scatter(&natural, &mut nout);
        assert_eq!(sout, nout);
    }

    #[test]
    fn huffman_nonzero_bitmap_matches_scalar() {
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

        // Sparse, including boundaries (k=0 DC, k=63 last AC, k=15/16
        // straddling the 16-lane chunk boundary).
        let mut block = [0i16; 64];
        for k in [0, 1, 7, 8, 15, 16, 31, 32, 47, 48, 62, 63] {
            block[k] = (k as i16) + 1;
        }
        assert_eq!(
            scalar::huffman::nonzero_bitmap(&block),
            huffman::nonzero_bitmap(&block),
        );

        // Extreme magnitudes (i16::MIN must register as nonzero too).
        let mut block = [0i16; 64];
        block[0] = i16::MIN;
        block[63] = i16::MAX;
        assert_eq!(
            scalar::huffman::nonzero_bitmap(&block),
            huffman::nonzero_bitmap(&block),
        );

        // Deterministic LCG panel.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let mut block = [0i16; 64];
        for v in block.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Allow zeros so the bitmap has varied bits.
            *v = ((state >> 55) as i16).wrapping_sub(128);
        }
        assert_eq!(
            scalar::huffman::nonzero_bitmap(&block),
            huffman::nonzero_bitmap(&block),
        );
    }

    fn idct_xcheck(coef: &[i16; 64], tag: &str) {
        let mut s = [0u8; 64];
        let mut n = [0u8; 64];
        scalar::dct::idct_islow(coef, &mut s);
        dct::idct_islow(coef, &mut n);
        assert_eq!(s, n, "{tag}");
    }

    #[test]
    fn idct_neon_matches_scalar_zeros() {
        idct_xcheck(&[0i16; 64], "zeros");
    }

    #[test]
    fn idct_neon_sparse_path_top_only() {
        // Blocks whose rows 4-7 are zero exercise `idct_pass1_sparse`
        // on both 4x8 halves. Use a panel of low-frequency AC values
        // covering all (row 0-3, col 0-7) positions in turn.
        for k in 0..32 {
            for &m in &[1i16, -1, 17, 256, -256, 1024] {
                let mut coef = [0i16; 64];
                coef[k] = m;
                idct_xcheck(&coef, &format!("top-only k={k} m={m}"));
            }
        }
    }

    #[test]
    fn idct_neon_sparse_path_right_half_zero() {
        // Blocks whose right 4x8 half (cols 4-7) is fully zero
        // exercise `idct_pass2_sparse`. Combine non-trivial left-half
        // content with zero right-half.
        for seed in 0..40u64 {
            let mut block = random_block(seed);
            for v in block.iter_mut() {
                *v /= 8;
            }
            // Run FDCT to get a coef-shaped distribution, then zero
            // cols 4-7.
            scalar::dct::fdct_islow(&mut block);
            for r in 0..8 {
                for c in 4..8 {
                    block[r * 8 + c] = 0;
                }
            }
            idct_xcheck(&block, &format!("right-zero seed={seed}"));
        }
    }

    #[test]
    fn idct_neon_matches_scalar_dc_only() {
        // DC swept across the libjpeg-turbo NEON IDCT input contract
        // (i12 raw range × typical small quant ⇒ values that keep
        // pass-1 intermediates inside i16). The ±2047 endpoints stress
        // the saturation path. This panel also exercises the DC-only
        // sparse fast path (rows 1-7 all zero).
        for &dc in &[2047i16, -2048, 1024, -1024, 512, -512, 8, -8] {
            let mut coef = [0i16; 64];
            coef[0] = dc;
            idct_xcheck(&coef, &format!("dc={dc}"));
        }
    }

    #[test]
    fn idct_neon_sparse_path_dc_only() {
        // Direct verification of the DC-only sparse path: dense DC
        // sweep including saturation extremes and zero. The sparse
        // detection must trigger for every input here (all AC == 0).
        for dc in -2048i16..=2047i16 {
            let mut coef = [0i16; 64];
            coef[0] = dc;
            // Compute scalar reference for this DC and compare. The
            // scalar produces a flat block: `clamp((dc+4)>>3 + 128, 0, 255)`.
            let expected = ((((dc as i32) + 4) >> 3) + 128).clamp(0, 255) as u8;
            let mut neon_out = [0u8; 64];
            dct::idct_islow(&coef, &mut neon_out);
            for (i, &v) in neon_out.iter().enumerate() {
                assert_eq!(v, expected, "dc={dc} pos={i}");
            }
        }
    }

    #[test]
    fn idct_neon_matches_scalar_ac_impulse() {
        // One AC coefficient at a time, magnitude swept through small
        // and large values within JPEG raw-coef range. Hits every odd /
        // even part butterfly path.
        for k in 1..64 {
            for &m in &[1i16, -1, 17, -17, 256, -256, 1024, -1024, 2047, -2048] {
                let mut coef = [0i16; 64];
                coef[k] = m;
                idct_xcheck(&coef, &format!("k={k} m={m}"));
            }
        }
    }

    #[test]
    fn idct_neon_matches_scalar_natural() {
        // Round-trip a forward-DCT'd natural-looking block: ramp plus
        // mild noise, then idct. Models the realistic decoder input
        // distribution (small AC, modest DC).
        for seed in 0..100u64 {
            let mut block = random_block(seed);
            // Reduce magnitude so post-FDCT values stay decoder-realistic.
            for v in block.iter_mut() {
                *v = (*v / 4).saturating_add(if seed.is_multiple_of(3) { 32 } else { 0 });
            }
            // FDCT in place to get a coef-shaped distribution.
            scalar::dct::fdct_islow(&mut block);
            idct_xcheck(&block, &format!("natural seed={seed}"));
        }
    }

    #[test]
    fn idct_neon_matches_scalar_dequantized_real_range() {
        // Simulate real decoder traffic: FDCT a natural-looking 8x8
        // block, quantize via the standard luma table at q=75, then
        // dequantize. The resulting block matches the post-dequant
        // distribution that the decoder feeds into the IDCT. This is
        // the input shape libjpeg-turbo's NEON kernel was tuned for.
        use crate::quant::build_divisors;
        use crate::tables::{STD_LUMA_QUANT, scale_quant_table};
        let qtab = scale_quant_table(&STD_LUMA_QUANT, 75);
        let div = build_divisors(&qtab);
        for seed in 0..50u64 {
            // Build a natural-looking source block: smooth gradient
            // plus low-amplitude noise.
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
            // Dequantize = element-wise multiply by quant table.
            let mut coef = [0i16; 64];
            for i in 0..64 {
                coef[i] = (quantized[i] as i32 * qtab[i] as i32).clamp(-32768, 32767) as i16;
            }
            idct_xcheck(&coef, &format!("dequant seed={seed}"));
        }
    }

    // ---- sample (fancy chroma upsample) cross-check ----

    fn lcg_bytes(seed: u64, n: usize) -> Vec<u8> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect()
    }

    #[test]
    fn sample_neon_h2v2_fancy_vblend_matches_scalar() {
        for &n in &[
            1usize, 2, 7, 15, 16, 17, 31, 32, 33, 47, 48, 64, 65, 128, 257, 1024,
        ] {
            for seed in 0..3u64 {
                let cur = lcg_bytes(seed.wrapping_mul(2), n);
                let nbr = lcg_bytes(seed.wrapping_mul(2).wrapping_add(1), n);
                let mut a = vec![0u8; n];
                let mut b = vec![0u8; n];
                scalar::sample::h2v2_fancy_vblend(&cur, &nbr, &mut a, n);
                sample::h2v2_fancy_vblend(&cur, &nbr, &mut b, n);
                assert_eq!(a, b, "n={n} seed={seed}");
            }
        }
    }

    #[test]
    fn sample_neon_h2v2_fancy_vblend_extremes() {
        for &n in &[16usize, 17, 32, 33, 48] {
            let zeros = vec![0u8; n];
            let max = vec![255u8; n];
            let mut a = vec![0u8; n];
            let mut b = vec![0u8; n];
            scalar::sample::h2v2_fancy_vblend(&zeros, &max, &mut a, n);
            sample::h2v2_fancy_vblend(&zeros, &max, &mut b, n);
            assert_eq!(a, b, "cur=0,nbr=255 n={n}");
            scalar::sample::h2v2_fancy_vblend(&max, &zeros, &mut a, n);
            sample::h2v2_fancy_vblend(&max, &zeros, &mut b, n);
            assert_eq!(a, b, "cur=255,nbr=0 n={n}");
            scalar::sample::h2v2_fancy_vblend(&max, &max, &mut a, n);
            sample::h2v2_fancy_vblend(&max, &max, &mut b, n);
            assert_eq!(a, b, "cur=255,nbr=255 n={n}");
        }
    }

    #[test]
    fn sample_neon_h2_fancy_upsample_matches_scalar() {
        for &n in &[
            1usize, 2, 3, 7, 15, 16, 17, 18, 19, 31, 32, 33, 34, 47, 48, 49, 64, 65, 128, 257, 1024,
        ] {
            for seed in 0..3u64 {
                let src = lcg_bytes(seed.wrapping_add(0x42), n);
                let mut a = vec![0u8; 2 * n];
                let mut b = vec![0u8; 2 * n];
                scalar::sample::h2_fancy_upsample(&src, &mut a, n);
                sample::h2_fancy_upsample(&src, &mut b, n);
                assert_eq!(a, b, "n={n} seed={seed}");
            }
        }
    }

    #[test]
    fn sample_neon_h2_fancy_upsample_extremes() {
        for &n in &[18usize, 19, 32, 33, 48] {
            let zeros = vec![0u8; n];
            let max = vec![255u8; n];
            let mut a = vec![0u8; 2 * n];
            let mut b = vec![0u8; 2 * n];
            scalar::sample::h2_fancy_upsample(&zeros, &mut a, n);
            sample::h2_fancy_upsample(&zeros, &mut b, n);
            assert_eq!(a, b, "zeros n={n}");
            scalar::sample::h2_fancy_upsample(&max, &mut a, n);
            sample::h2_fancy_upsample(&max, &mut b, n);
            assert_eq!(a, b, "max n={n}");
            // Edge-clamp stressor: a single spike at index 0 and n-1.
            let mut spike = vec![0u8; n];
            spike[0] = 255;
            spike[n - 1] = 255;
            scalar::sample::h2_fancy_upsample(&spike, &mut a, n);
            sample::h2_fancy_upsample(&spike, &mut b, n);
            assert_eq!(a, b, "edge spike n={n}");
        }
    }

    #[test]
    #[ignore = "kernel micro-bench, run with --ignored --nocapture"]
    fn sample_neon_micro_bench() {
        use std::time::Instant;
        let n = 4096;
        let iters = 50_000;
        let src = lcg_bytes(0xA5A5_A5A5, n);
        let nbr = lcg_bytes(0x5A5A_5A5A, n);
        let mut out = vec![0u8; n];

        // h2v2_fancy_vblend
        let t = Instant::now();
        for _ in 0..iters {
            scalar::sample::h2v2_fancy_vblend(&src, &nbr, &mut out, n);
            std::hint::black_box(&out);
        }
        let s = t.elapsed().as_nanos() as f64 / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            sample::h2v2_fancy_vblend(&src, &nbr, &mut out, n);
            std::hint::black_box(&out);
        }
        let nv = t.elapsed().as_nanos() as f64 / iters as f64;
        println!(
            "  h2v2_fancy_vblend n={n}: scalar={:.2}us neon={:.2}us speedup={:.2}x",
            s / 1000.0,
            nv / 1000.0,
            s / nv
        );

        // h2_fancy_upsample
        let mut up = vec![0u8; 2 * n];
        let t = Instant::now();
        for _ in 0..iters {
            scalar::sample::h2_fancy_upsample(&src, &mut up, n);
            std::hint::black_box(&up);
        }
        let s = t.elapsed().as_nanos() as f64 / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            sample::h2_fancy_upsample(&src, &mut up, n);
            std::hint::black_box(&up);
        }
        let nv = t.elapsed().as_nanos() as f64 / iters as f64;
        println!(
            "  h2_fancy_upsample n={n}: scalar={:.2}us neon={:.2}us speedup={:.2}x",
            s / 1000.0,
            nv / 1000.0,
            s / nv
        );
    }
}
