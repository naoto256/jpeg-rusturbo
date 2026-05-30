//! Grayscale (1-component) encode + decode integration tests.
//!
//! Covers the new `JpegEncoder::encode_grayscale` entry point and the
//! `PixelFormat::Gray` decode-output path. Mirrors the structure of
//! `tests/roundtrip.rs` (gradient inputs, PSNR / max-diff floors) and
//! `tests/comparison_progressive.rs` (cross-decoder agreement with the
//! `image` crate).

use image::{ImageFormat, ImageReader};
use jpeg_rusturbo::decode::Decoder;
use jpeg_rusturbo::{JpegEncoder, PixelFormat};
use std::io::Cursor;

/// Smoothly-varying grayscale image — well-suited to JPEG and tolerant
/// of the quantization round-trip without needing aggressive tolerances.
fn gradient_gray(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            // Diagonal ramp 0..255 — gentle gradient.
            let v = (((x + y) * 255) / (w + h).max(1)) as u8;
            buf.push(v);
        }
    }
    buf
}

/// Same gradient but interleaved into an RGB buffer where R=G=B=Y.
/// Encoding this through `encode_rgb` then asking the decoder for
/// `PixelFormat::Gray` should reproduce the original Y plane (within
/// the quantization round-trip — RGB→YCbCr on R=G=B inputs is exact
/// luma-wise but the 4:2:0 chroma round-trip still gates total error).
fn gradient_gray_as_rgb(w: u32, h: u32) -> Vec<u8> {
    let gray = gradient_gray(w, h);
    let mut buf = Vec::with_capacity(gray.len() * 3);
    for &v in &gray {
        buf.extend_from_slice(&[v, v, v]);
    }
    buf
}

fn max_abs_diff(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x as i32) - (*y as i32)).abs())
        .max()
        .unwrap_or(0)
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| ((*x as i32) - (*y as i32)).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

fn encode_gray(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let mut jpeg = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg, quality);
    enc.encode_grayscale(pixels, w, h)
        .expect("encode_grayscale");
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI");
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "EOI");
    jpeg
}

/// Round-trip: encode gray → decode gray → assert luma is faithful.
fn run_gray_roundtrip(w: u32, h: u32, quality: u8, max_diff: i32) {
    let gray = gradient_gray(w, h);
    let jpeg = encode_gray(&gray, w, h, quality);

    // Header: must be a 1-component frame.
    let dec = Decoder::new(&jpeg).expect("our decoder headers");
    let info = dec.info();
    assert_eq!(info.width, w);
    assert_eq!(info.height, h);
    assert_eq!(info.components, 1, "expected 1-component JPEG");

    let decoded = dec.decode(PixelFormat::Gray).expect("our decode → Gray");
    assert_eq!(decoded.len(), (w * h) as usize);
    let d = max_abs_diff(&gray, &decoded);
    assert!(
        d <= max_diff,
        "gray roundtrip max diff {} > {} (q={}, {}x{})",
        d,
        max_diff,
        quality,
        w,
        h
    );
}

#[test]
fn gray_roundtrip_8x8() {
    run_gray_roundtrip(8, 8, 80, 8);
}

#[test]
fn gray_roundtrip_17x17_unaligned() {
    run_gray_roundtrip(17, 17, 80, 8);
}

#[test]
fn gray_roundtrip_1592x1124() {
    run_gray_roundtrip(1592, 1124, 80, 8);
}

#[test]
fn gray_roundtrip_high_quality() {
    // q=95 should round-trip even tighter.
    run_gray_roundtrip(64, 64, 95, 4);
}

/// Encoded grayscale JPEG must decode to RGB with R=G=B=Y (the decoder
/// already replicates Y across all channels for 1-component sources;
/// this test locks that behaviour down).
#[test]
fn gray_jpeg_decodes_to_rgb_with_equal_channels() {
    let (w, h) = (32, 32);
    let gray = gradient_gray(w, h);
    let jpeg = encode_gray(&gray, w, h, 90);

    let rgb = jpeg_rusturbo::decode::decode(&jpeg, PixelFormat::Rgb).expect("decode → RGB");
    assert_eq!(rgb.len(), (w * h * 3) as usize);
    for px in rgb.chunks_exact(3) {
        assert_eq!(px[0], px[1], "R != G in gray→RGB decode");
        assert_eq!(px[1], px[2], "G != B in gray→RGB decode");
    }
}

/// Decode a color (3-component) JPEG into `PixelFormat::Gray` and
/// assert the result matches the input's luma (the Y plane the
/// encoder built from RGB).
#[test]
fn color_jpeg_decodes_to_gray_as_luma() {
    let (w, h) = (64, 64);
    let rgb = gradient_gray_as_rgb(w, h);
    // R=G=B=Y so the BT.601 luma the encoder builds equals the input
    // ramp byte-for-byte (modulo the quantization round-trip).
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 90)
        .encode_rgb(&rgb, w, h)
        .expect("encode_rgb");

    let gray_out = jpeg_rusturbo::decode::decode(&jpeg, PixelFormat::Gray).expect("decode → Gray");
    assert_eq!(gray_out.len(), (w * h) as usize);
    let want = gradient_gray(w, h);
    let mean = mean_abs_diff(&want, &gray_out);
    assert!(
        mean < 2.0,
        "color → Gray luma mean abs diff {} too large",
        mean
    );
}

/// Cross-decoder check: our grayscale JPEG round-trips through the
/// `image` crate's decoder (zune-jpeg in image 0.25) with agreement
/// well inside the JPEG quantization round-trip.
#[test]
fn cross_decoder_agreement_with_image_crate() {
    let (w, h) = (256, 192);
    let gray = gradient_gray(w, h);
    let jpeg = encode_gray(&gray, w, h, 85);

    let dyn_img = ImageReader::with_format(Cursor::new(&jpeg), ImageFormat::Jpeg)
        .decode()
        .expect("image::decode");
    let luma = dyn_img.to_luma8();
    assert_eq!(luma.width(), w);
    assert_eq!(luma.height(), h);
    let ours = jpeg_rusturbo::decode::decode(&jpeg, PixelFormat::Gray).expect("our decode → Gray");
    let theirs = luma.into_raw();
    assert_eq!(ours.len(), theirs.len());

    let max = max_abs_diff(&ours, &theirs);
    assert!(
        max <= 1,
        "cross-decoder per-pixel max diff {} > 1 (we should agree on a JPEG neither of us authored beyond clamp noise)",
        max
    );
}

/// `set_optimize_huffman(true) + encode_grayscale(...)`: must produce
/// a strictly smaller output AND decode to the same pixels as the
/// standard-tables path.
#[test]
fn optimize_huffman_composes() {
    let (w, h) = (512, 384);
    let gray = gradient_gray(w, h);

    let std_jpeg = encode_gray(&gray, w, h, 80);

    let mut opt_jpeg = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut opt_jpeg, 80);
        enc.set_optimize_huffman(true);
        enc.encode_grayscale(&gray, w, h)
            .expect("encode_grayscale optimize");
    }
    assert!(
        opt_jpeg.len() < std_jpeg.len(),
        "optimize-Huffman grayscale ({} B) should be smaller than standard ({} B)",
        opt_jpeg.len(),
        std_jpeg.len()
    );

    let dec_std = jpeg_rusturbo::decode::decode(&std_jpeg, PixelFormat::Gray).unwrap();
    let dec_opt = jpeg_rusturbo::decode::decode(&opt_jpeg, PixelFormat::Gray).unwrap();
    assert_eq!(
        dec_std, dec_opt,
        "optimized vs standard Huffman should decode to identical pixels"
    );
}

/// Progressive grayscale is explicitly unsupported in this version —
/// must error with `Unsupported` rather than silently producing a
/// broken stream.
#[test]
fn progressive_grayscale_rejected() {
    let mut out = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
    enc.set_progressive(true);
    let err = enc
        .encode_grayscale(&vec![128u8; 16 * 16], 16, 16)
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

/// `set_threads(n)` on grayscale must be a no-op (no thread fan-out
/// yet) — output is identical to the serial path regardless of `n`.
#[test]
fn threads_setter_is_noop_for_grayscale() {
    let (w, h) = (200, 150);
    let gray = gradient_gray(w, h);

    let mut serial = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut serial, 80);
        enc.set_threads(1);
        enc.encode_grayscale(&gray, w, h).unwrap();
    }
    let mut threaded = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut threaded, 80);
        enc.set_threads(4);
        enc.encode_grayscale(&gray, w, h).unwrap();
    }
    assert_eq!(
        serial, threaded,
        "grayscale output must be identical regardless of set_threads(...)"
    );
}

/// File-size sanity: on noisy content where the luma plane carries
/// real entropy, a 1-component grayscale JPEG should be smaller than
/// the same content re-encoded through the 4:2:0 RGB path (the latter
/// pays for a full chroma DQT + DHT and two chroma planes of entropy
/// coding plus the chroma DC term per MCU).
///
/// We use a pseudo-noise pattern instead of a smooth gradient because
/// R=G=B smooth ramps produce chroma planes that compress to almost
/// nothing — that case understates the gray-vs-color delta. The
/// README documents the headline "roughly a third" on realistic
/// photographic content; this test asserts only `gray < color`.
#[test]
fn grayscale_file_size_smaller_than_color() {
    let (w, h) = (512, 512);
    // Linear-congruential noise — deterministic, content-dense.
    let mut gray = Vec::with_capacity((w * h) as usize);
    let mut s: u32 = 0x1234_5678;
    for _ in 0..(w * h) {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        gray.push((s >> 16) as u8);
    }
    let mut rgb = Vec::with_capacity(gray.len() * 3);
    for &v in &gray {
        rgb.extend_from_slice(&[v, v, v]);
    }

    let gray_jpeg = encode_gray(&gray, w, h, 80);
    let mut color_jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut color_jpeg, 80)
        .encode_rgb(&rgb, w, h)
        .unwrap();

    assert!(
        gray_jpeg.len() < color_jpeg.len(),
        "expected gray ({} B) < color-4:2:0 ({} B)",
        gray_jpeg.len(),
        color_jpeg.len()
    );
}

/// `encode` (generic) with `PixelFormat::Gray` must produce
/// byte-identical output to `encode_grayscale`.
#[test]
fn generic_encode_gray_matches_convenience() {
    let (w, h) = (40, 24);
    let gray = gradient_gray(w, h);

    let mut a = Vec::new();
    JpegEncoder::new_with_quality(&mut a, 80)
        .encode_grayscale(&gray, w, h)
        .unwrap();
    let mut b = Vec::new();
    JpegEncoder::new_with_quality(&mut b, 80)
        .encode(&gray, w, h, PixelFormat::Gray)
        .unwrap();
    assert_eq!(a, b);
}
