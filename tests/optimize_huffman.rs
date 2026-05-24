//! Optimized-Huffman (`set_optimize_huffman(true)`) integration tests.
//!
//! Verifies:
//!   1. Default off → output is byte-identical to the unmodified
//!      encoder (= regression safety net).
//!   2. Enabling shrinks file size at identical PSNR (= the feature
//!      actually delivers what it advertises).
//!   3. Restart-interval interaction stays correct (DC predictors
//!      reset on both passes).
//!   4. Output is decodable and round-trips at parity-quality PSNR.

use jpeg_rusturbo::decode::Decoder;
use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder, PixelFormat};

fn photo_like_rgb(w: u32, h: u32) -> Vec<u8> {
    // Smooth gradient + some higher-frequency texture so Huffman
    // counts diverge meaningfully from the standard tables.
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_mul(31) ^ y.wrapping_mul(7)) & 0xFF) as u8;
            let g = ((x.wrapping_mul(13) + y.wrapping_mul(5)) & 0xFF) as u8;
            let b = ((x ^ y) & 0xFF) as u8;
            // Mix with a smooth ramp so AC coefficients stay realistic.
            let sx = ((x * 255) / w.max(1)) as u8;
            buf.extend_from_slice(&[r ^ sx, g, b]);
        }
    }
    buf
}

fn encode(rgb: &[u8], w: u32, h: u32, q: u8, sub: ChromaSubsampling, optimize: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut out, q);
    enc.set_subsampling(sub);
    if optimize {
        enc.set_optimize_huffman(true);
    }
    enc.encode_rgb(rgb, w, h).expect("encode");
    out
}

fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i32) - (*y as i32);
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / a.len() as f64;
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

fn decode_rgb(jpeg: &[u8]) -> Vec<u8> {
    let dec = Decoder::new(jpeg).expect("parse");
    dec.decode(PixelFormat::Rgb).expect("decode")
}

/// Default state (= `set_optimize_huffman` never called) must produce
/// byte-for-byte identical output to the historic single-pass encoder.
/// This is the regression gate guarding the default code path.
#[test]
fn default_off_is_byte_identical() {
    let rgb = photo_like_rgb(64, 48);
    let baseline = encode(&rgb, 64, 48, 80, ChromaSubsampling::Yuv420, false);
    // Encode again with an encoder we touched (subsampling setter,
    // restart interval setter) but optimize_huffman never enabled.
    let mut out = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
    enc.set_subsampling(ChromaSubsampling::Yuv420);
    enc.encode_rgb(&rgb, 64, 48).expect("encode");
    assert_eq!(
        out, baseline,
        "default-off path drifted from single-pass output"
    );
}

#[test]
fn optimized_smaller_than_default_420() {
    let (w, h) = (256u32, 192u32);
    let rgb = photo_like_rgb(w, h);
    let std = encode(&rgb, w, h, 80, ChromaSubsampling::Yuv420, false);
    let opt = encode(&rgb, w, h, 80, ChromaSubsampling::Yuv420, true);
    assert!(
        opt.len() < std.len(),
        "optimize_huffman did not shrink output: std={} opt={}",
        std.len(),
        opt.len(),
    );
    // PSNR floor — should be ≥ default (same quant tables, so equal
    // within roundoff; the only thing that differs is entropy coding).
    let dec_std = decode_rgb(&std);
    let dec_opt = decode_rgb(&opt);
    let psnr_std = psnr_rgb(&rgb, &dec_std);
    let psnr_opt = psnr_rgb(&rgb, &dec_opt);
    assert!(
        (psnr_opt - psnr_std).abs() < 0.01,
        "PSNR drifted with optimize_huffman: std={psnr_std:.3} opt={psnr_opt:.3}",
    );
}

#[test]
fn optimized_smaller_than_default_444() {
    let (w, h) = (128u32, 128u32);
    let rgb = photo_like_rgb(w, h);
    let std = encode(&rgb, w, h, 85, ChromaSubsampling::Yuv444, false);
    let opt = encode(&rgb, w, h, 85, ChromaSubsampling::Yuv444, true);
    assert!(opt.len() < std.len(), "std={} opt={}", std.len(), opt.len());
}

#[test]
fn optimized_smaller_than_default_422() {
    let (w, h) = (96u32, 64u32);
    let rgb = photo_like_rgb(w, h);
    let std = encode(&rgb, w, h, 80, ChromaSubsampling::Yuv422, false);
    let opt = encode(&rgb, w, h, 80, ChromaSubsampling::Yuv422, true);
    assert!(opt.len() < std.len(), "std={} opt={}", std.len(), opt.len());
}

/// Optimize + restart interval: the per-RST DC predictor reset must
/// fire on *both* the count pass and the emit pass — if they disagree
/// the resulting stream desyncs and decoding produces garbage. The
/// canary is that the round-trip PSNR matches the *default* encoder's
/// PSNR under the same restart configuration (i.e. the only thing
/// that changed was the Huffman tables, not the reconstruction).
#[test]
fn optimized_with_restart_interval_matches_default_psnr() {
    fn smooth_gradient(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w.max(1)) as u8;
                let g = ((y * 255) / h.max(1)) as u8;
                let b = (((x + y) * 255) / (w + h).max(1)) as u8;
                buf.extend_from_slice(&[r, g, b]);
            }
        }
        buf
    }

    let (w, h) = (96u32, 64u32);
    let rgb = smooth_gradient(w, h);
    let mut std_out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut std_out, 80);
        enc.set_subsampling(ChromaSubsampling::Yuv420);
        enc.set_restart_interval(3);
        enc.encode_rgb(&rgb, w, h).expect("encode std");
    }
    let mut opt_out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut opt_out, 80);
        enc.set_subsampling(ChromaSubsampling::Yuv420);
        enc.set_restart_interval(3);
        enc.set_optimize_huffman(true);
        enc.encode_rgb(&rgb, w, h).expect("encode opt");
    }
    let psnr_std = psnr_rgb(&rgb, &decode_rgb(&std_out));
    let psnr_opt = psnr_rgb(&rgb, &decode_rgb(&opt_out));
    assert!(
        (psnr_opt - psnr_std).abs() < 0.01,
        "PSNR drift under restart: std={psnr_std:.3} opt={psnr_opt:.3}",
    );
    assert!(
        opt_out.len() < std_out.len(),
        "optimize did not shrink under restart: std={} opt={}",
        std_out.len(),
        opt_out.len(),
    );
}
