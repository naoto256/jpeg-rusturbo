//! Micro-benchmark: time `JpegEncoder::encode_rgba` over a few
//! representative resolutions. Prints per-iteration milliseconds and a
//! crude ms/MPx figure so we can compare runs at a glance.
//!
//! Build with `cargo run -p jpeg-rusturbo --release --bin bench`. The
//! build label reflects the active arch backend selected at compile time.

use std::time::Instant;

use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder};

const ITERATIONS: usize = 100;

fn main() {
    println!("jpeg-rusturbo bench — {ITERATIONS} iterations per resolution");
    println!(
        "build: {arch}, profile: {profile}",
        arch = if cfg!(all(target_arch = "aarch64", not(feature = "force-scalar"))) {
            "aarch64 (NEON kernels enabled)"
        } else if cfg!(target_arch = "aarch64") {
            "aarch64 (force-scalar — NEON kernels disabled)"
        } else if cfg!(all(target_arch = "x86_64", not(feature = "force-scalar"))) {
            "x86_64 (AVX2: quant+dct+color+downsample; scalar: huffman)"
        } else if cfg!(target_arch = "x86_64") {
            "x86_64 (force-scalar — AVX2 kernels disabled)"
        } else {
            "scalar (other arch)"
        },
        profile = if cfg!(debug_assertions) {
            "debug (numbers will be useless)"
        } else {
            "release"
        },
    );

    for &(mode_label, mode) in &[
        ("4:2:0", ChromaSubsampling::Yuv420),
        ("4:2:2", ChromaSubsampling::Yuv422),
    ] {
        println!("\nsubsampling: {mode_label}");
        for &(label, w, h) in &[
            ("1592x1124 (session-size)", 1592u32, 1124u32),
            ("1920x1080 (1080p)", 1920, 1080),
            ("3840x2160 (4K)", 3840, 2160),
        ] {
            bench_one(label, w, h, mode);
        }
    }
}

fn bench_one(label: &str, w: u32, h: u32, subsampling: ChromaSubsampling) {
    let pixels = make_image(w, h);
    let mut buf = Vec::with_capacity((w as usize * h as usize * 3) / 2);

    // Warm-up.
    for _ in 0..3 {
        buf.clear();
        let mut enc = JpegEncoder::new_with_quality(&mut buf, 80);
        enc.set_subsampling(subsampling);
        enc.encode_rgba(&pixels, w, h).unwrap();
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        buf.clear();
        let mut enc = JpegEncoder::new_with_quality(&mut buf, 80);
        enc.set_subsampling(subsampling);
        enc.encode_rgba(&pixels, w, h).unwrap();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / ITERATIONS as u32;
    let mp = (w as f64) * (h as f64) / 1_000_000.0;
    let ms_per_mp = per_iter.as_secs_f64() * 1000.0 / mp;
    println!(
        "  {label:<28}  {ms:>7.2} ms/iter   {ms_per_mp:>5.2} ms/MPx   ({size} bytes)",
        ms = per_iter.as_secs_f64() * 1000.0,
        size = buf.len(),
    );
}

/// Synthesize a deterministic test image. Smooth-ish gradients with a
/// sprinkling of high-frequency detail so the encoder isn't degenerate.
fn make_image(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x ^ y) & 0xFF) as u8;
            let g = ((x.wrapping_add(y)) & 0xFF) as u8;
            let b = (((x.wrapping_mul(7)) ^ (y.wrapping_mul(13))) & 0xFF) as u8;
            v.push(r);
            v.push(g);
            v.push(b);
            v.push(255);
        }
    }
    v
}
