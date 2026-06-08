//! Comparison harness vs the `image` crate (jpeg-decoder under the
//! hood for decode; image's own encoder for encode).
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench vs_image
//! ```
//!
//! A plain `fn main()` printer (harness disabled in `Cargo.toml`) rather
//! than a libtest bench — the output is intentionally human-readable
//! plaintext that we cite in BENCH.md and the README.

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

/// Natural-like content: smooth sky region (DC-dominant blocks),
/// a textured mid band (low-AC), and bottom solid bars (DC blocks
/// with strong edges between them). Mirrors `make_natural_image` in
/// `benches/pipeline.rs` but emits 3-byte RGB (no alpha) so it can feed
/// `encode_rgb` directly. See `BENCH.md` Section D-natural for the
/// design rationale.
fn make_natural_image(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    let smooth_h = (h as u64 * 70) / 100;
    let texture_h = (h as u64 * 20) / 100;
    for y in 0..h {
        let yu = y as u64;
        for x in 0..w {
            let xu = x as u64;
            let (r, g, b);
            if yu < smooth_h {
                let t = (yu * 80 / smooth_h.max(1)) as u8;
                r = 130u8.saturating_sub(t / 2);
                g = 160u8.saturating_sub(t / 3);
                b = 200u8.saturating_sub(t / 4);
            } else if yu < smooth_h + texture_h {
                let n = (xu.wrapping_mul(2654435761) ^ yu.wrapping_mul(40503)) & 0x0F;
                let base = 110u8;
                r = base + (n as u8);
                g = base + (((n * 3) & 0x0F) as u8);
                b = base + (((n * 5) & 0x0F) as u8);
            } else {
                let bar = (xu / 64) & 1;
                if bar == 0 {
                    r = 40;
                    g = 60;
                    b = 70;
                } else {
                    r = 230;
                    g = 220;
                    b = 200;
                }
            }
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

fn encode_with_ours(
    rgb: &[u8],
    w: u32,
    h: u32,
    q: u8,
    sub: ChromaSubsampling,
    threads: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity((w as usize) * (h as usize));
    let mut enc = OurEncoder::new_with_quality(&mut out, q);
    enc.set_subsampling(sub);
    enc.set_threads(threads);
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

fn main() {
    let cases = [
        ("1592x1124 (session)", 1592u32, 1124u32),
        ("1920x1080 (1080p)", 1920, 1080),
        ("3840x2160 (4K)", 3840, 2160),
    ];

    println!();
    println!(
        "jpeg-rusturbo vs image — {} iters per case (mean of single timed batch)",
        ITER
    );

    for &(thread_label, threads) in &[("threads=1", 1u32), ("threads=auto", 0u32)] {
        // `image` crate's JPEG encoder is hardcoded to 4:2:0; we vary our side only
        // for completeness. Ratio is meaningful only on the 4:2:0 row.
        for &(sub_label, sub) in &[
            ("4:4:4", ChromaSubsampling::Yuv444),
            ("4:2:2", ChromaSubsampling::Yuv422),
            ("4:2:0", ChromaSubsampling::Yuv420),
        ] {
            println!();
            println!("=== Encode (q=80, {thread_label}, ours={sub_label} | image=4:2:0 fixed) ===");
            println!(
                "{:<24}  {:>12}  {:>12}  {:>7}",
                "case", "ours (us)", "image (us)", "ratio"
            );
            for &(label, w, h) in &cases {
                let rgb = make_image(w, h);
                let ours_us = time_us(|| {
                    let buf = encode_with_ours(&rgb, w, h, 80, sub, threads);
                    std::hint::black_box(buf);
                });
                let theirs_us = time_us(|| {
                    let buf = encode_with_image(&rgb, w, h, 80);
                    std::hint::black_box(buf);
                });
                let ratio = theirs_us / ours_us;
                println!("{label:<24}  {ours_us:>12.1}  {theirs_us:>12.1}  {ratio:>5.2}x",);
            }
        }
    }

    // Decode is run on both corpora. Synthetic XOR is the
    // Huffman-heavy worst case (every block is full-AC, sparse and
    // SWAR-refill gains barely register). Natural-like content
    // mixes DC-dominant sky, low-AC texture, and edge bars — closer
    // to realistic web/photo input and where 0.7.0's AVX2 sparse
    // parity + SWAR refill actually pay off.
    for &(corpus_label, corpus_fn) in &[
        (
            "synthetic (worst-case)",
            make_image as fn(u32, u32) -> Vec<u8>,
        ),
        (
            "natural content",
            make_natural_image as fn(u32, u32) -> Vec<u8>,
        ),
    ] {
        println!();
        println!("=== Decode (q=80, 4:2:0 JPEGs produced by our encoder, {corpus_label}) ===");
        println!(
            "{:<24}  {:>12}  {:>12}  {:>7}",
            "case", "ours (us)", "image (us)", "ratio"
        );
        for &(label, w, h) in &cases {
            let rgb = corpus_fn(w, h);
            let jpeg = encode_with_ours(&rgb, w, h, 80, ChromaSubsampling::Yuv420, 1);
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
    }
    println!();
    println!("(ratio > 1 means jpeg-rusturbo is faster)");
}
