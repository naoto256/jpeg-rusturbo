//! Comparison harness vs the `image` crate (jpeg-decoder under the
//! hood for decode; image's own encoder for encode).
//!
//! Run with:
//!
//! ```text
//! cargo test --release --test comparison_bench -- --nocapture comparison_print
//! ```
//!
//! The output is intentionally human-readable plaintext rather than
//! the usual `assert_*!` test failure mode — this is the harness we
//! cite in BENCH.md and the README.

use image::ImageEncoder;
use image::codecs::jpeg::JpegEncoder as ImageJpegEncoder;
use image::{ImageFormat, ImageReader};
use jpeg_rusturbo::decode::Decoder as OurDecoder;
use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder as OurEncoder, PixelFormat};
use std::io::Cursor;
use std::time::Instant;

const ITER: usize = 50;

fn make_image(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x ^ y) & 0xFF) as u8;
            let g = ((x.wrapping_add(y)) & 0xFF) as u8;
            let b = (((x.wrapping_mul(7)) ^ (y.wrapping_mul(13))) & 0xFF) as u8;
            v.extend_from_slice(&[r, g, b]);
        }
    }
    v
}

fn time_us<F: FnMut()>(mut f: F) -> f64 {
    // Warm up.
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITER {
        f();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1e6 / ITER as f64
}

fn encode_with_ours(rgb: &[u8], w: u32, h: u32, q: u8, sub: ChromaSubsampling) -> Vec<u8> {
    let mut out = Vec::with_capacity((w as usize) * (h as usize));
    let mut enc = OurEncoder::new_with_quality(&mut out, q);
    enc.set_subsampling(sub);
    enc.encode_rgb(rgb, w, h).unwrap();
    out
}

fn encode_with_image(rgb: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity((w as usize) * (h as usize));
    let enc = ImageJpegEncoder::new_with_quality(&mut out, q);
    enc.write_image(rgb, w, h, image::ExtendedColorType::Rgb8)
        .unwrap();
    out
}

fn decode_with_ours(jpeg: &[u8]) -> Vec<u8> {
    OurDecoder::new(jpeg)
        .unwrap()
        .decode(PixelFormat::Rgb)
        .unwrap()
}

fn decode_with_image(jpeg: &[u8]) -> Vec<u8> {
    let img = ImageReader::with_format(Cursor::new(jpeg), ImageFormat::Jpeg)
        .decode()
        .unwrap()
        .to_rgb8();
    img.into_raw()
}

#[test]
fn comparison_print() {
    let cases = [
        ("1592x1124 (session)", 1592u32, 1124u32),
        ("1920x1080 (1080p)", 1920, 1080),
        ("3840x2160 (4K)", 3840, 2160),
    ];

    println!();
    println!(
        "jpeg-rusturbo vs image — {} iters per case (median of single timed batch)",
        ITER
    );
    println!();
    println!("=== Encode (q=80, 4:2:0) ===");
    println!(
        "{:<24}  {:>12}  {:>12}  {:>7}",
        "case", "ours (us)", "image (us)", "ratio"
    );
    for &(label, w, h) in &cases {
        let rgb = make_image(w, h);
        let ours_us = time_us(|| {
            let buf = encode_with_ours(&rgb, w, h, 80, ChromaSubsampling::Yuv420);
            std::hint::black_box(buf);
        });
        let theirs_us = time_us(|| {
            let buf = encode_with_image(&rgb, w, h, 80);
            std::hint::black_box(buf);
        });
        let ratio = theirs_us / ours_us;
        println!("{label:<24}  {ours_us:>12.1}  {theirs_us:>12.1}  {ratio:>5.2}x",);
    }

    println!();
    println!("=== Decode (q=80, 4:2:0 JPEGs produced by our encoder) ===");
    println!(
        "{:<24}  {:>12}  {:>12}  {:>7}",
        "case", "ours (us)", "image (us)", "ratio"
    );
    for &(label, w, h) in &cases {
        let rgb = make_image(w, h);
        let jpeg = encode_with_ours(&rgb, w, h, 80, ChromaSubsampling::Yuv420);
        let ours_us = time_us(|| {
            let pixels = decode_with_ours(&jpeg);
            std::hint::black_box(pixels);
        });
        let theirs_us = time_us(|| {
            let pixels = decode_with_image(&jpeg);
            std::hint::black_box(pixels);
        });
        let ratio = theirs_us / ours_us;
        println!("{label:<24}  {ours_us:>12.1}  {theirs_us:>12.1}  {ratio:>5.2}x",);
    }
    println!();
    println!("(ratio > 1 means jpeg-rusturbo is faster)");
}
