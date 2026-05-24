//! Threaded-encode tests.
//!
//! Two things to guarantee:
//!
//! 1. The encoded byte stream is independent of how many worker
//!    threads the front half uses. The DC predictor chain is
//!    MCU-ordered and the bit-stream is serial; we only move pure
//!    functions out of the loop body, so the output must be
//!    bit-identical across thread counts.
//! 2. `set_threads(1)` is byte-for-byte identical to a build that
//!    doesn't touch the knob at all — i.e. the default path is
//!    untouched.
//!
//! Speedup is exercised by `bench_threads_speedup_4k`, gated behind
//! `#[ignore]` so it doesn't run in the default `cargo test` (it
//! needs `--release` to be meaningful and depends on host core count).

use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder, PixelFormat};

fn gradient_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            buf.extend_from_slice(&[r, g, b, 255]);
        }
    }
    buf
}

fn encode(
    pixels: &[u8],
    w: u32,
    h: u32,
    quality: u8,
    sub: ChromaSubsampling,
    threads: Option<u32>,
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut out, quality);
    enc.set_subsampling(sub);
    if let Some(n) = threads {
        enc.set_threads(n);
    }
    enc.encode(pixels, w, h, PixelFormat::Rgba).unwrap();
    out
}

fn assert_identical_across_threads(w: u32, h: u32, quality: u8, sub: ChromaSubsampling) {
    let pixels = gradient_rgba(w, h);
    // Baseline: default (no set_threads call). This is the "untouched"
    // path and must match `set_threads(1)`.
    let baseline = encode(&pixels, w, h, quality, sub, None);
    for n in [1u32, 2, 3, 4, 0] {
        let got = encode(&pixels, w, h, quality, sub, Some(n));
        assert_eq!(
            baseline.len(),
            got.len(),
            "length differs at threads={n}, sub={sub:?}"
        );
        assert_eq!(baseline, got, "byte mismatch at threads={n}, sub={sub:?}");
    }
}

#[test]
fn threaded_matches_serial_420_512x256_q80() {
    assert_identical_across_threads(512, 256, 80, ChromaSubsampling::Yuv420);
}

#[test]
fn threaded_matches_serial_422_512x256_q80() {
    assert_identical_across_threads(512, 256, 80, ChromaSubsampling::Yuv422);
}

#[test]
fn threaded_matches_serial_444_512x256_q80() {
    assert_identical_across_threads(512, 256, 80, ChromaSubsampling::Yuv444);
}

#[test]
fn threaded_matches_serial_420_odd_dims_q80() {
    // Non-MCU-aligned dimensions exercise the padding logic in
    // `color::extract_mcu_*`. Picked an odd width and height that
    // don't divide MCU_W=16 / MCU_H=16.
    assert_identical_across_threads(517, 263, 80, ChromaSubsampling::Yuv420);
}

#[test]
fn threaded_matches_serial_with_restart_420() {
    // Restart markers reset DC predictors mid-scan; the threaded path
    // has to preserve that bookkeeping precisely.
    let w = 256;
    let h = 128;
    let pixels = gradient_rgba(w, h);
    let baseline = {
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_subsampling(ChromaSubsampling::Yuv420);
        enc.set_restart_interval(7);
        enc.encode(&pixels, w, h, PixelFormat::Rgba).unwrap();
        out
    };
    for n in [2u32, 4, 0] {
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_subsampling(ChromaSubsampling::Yuv420);
        enc.set_restart_interval(7);
        enc.set_threads(n);
        enc.encode(&pixels, w, h, PixelFormat::Rgba).unwrap();
        assert_eq!(baseline, out, "byte mismatch at threads={n} with restart=7");
    }
}

#[test]
#[ignore = "host-dependent speedup measurement; run with --release --ignored"]
fn bench_threads_speedup_1080p() {
    bench_speedup(1920, 1080, "1080p");
}

#[test]
#[ignore = "host-dependent speedup measurement; run with --release --ignored"]
fn bench_threads_speedup_4k() {
    bench_speedup(3840, 2160, "4K");
}

fn bench_speedup(w: u32, h: u32, label: &str) {
    use std::time::Instant;

    let pixels = gradient_rgba(w, h);
    let q = 80u8;
    let sub = ChromaSubsampling::Yuv420;

    let measure = |threads: Option<u32>| -> std::time::Duration {
        // Warm cache + JIT-equivalent steady state with one untimed run.
        let _ = encode(&pixels, w, h, q, sub, threads);
        let n = 3;
        let mut total = std::time::Duration::ZERO;
        for _ in 0..n {
            let t = Instant::now();
            let _ = encode(&pixels, w, h, q, sub, threads);
            total += t.elapsed();
        }
        total / n
    };

    let t1 = measure(Some(1));
    let t4 = measure(Some(4));
    let t_auto = measure(Some(0));

    let speedup4 = t1.as_secs_f64() / t4.as_secs_f64();
    let speedup_auto = t1.as_secs_f64() / t_auto.as_secs_f64();
    println!(
        "{label} q{q} 4:2:0: threads=1 {:.1}ms → threads=4 {:.1}ms ({speedup4:.2}× speedup)",
        t1.as_secs_f64() * 1e3,
        t4.as_secs_f64() * 1e3,
    );
    println!(
        "{label} q{q} 4:2:0: threads=1 {:.1}ms → threads=auto {:.1}ms ({speedup_auto:.2}× speedup)",
        t1.as_secs_f64() * 1e3,
        t_auto.as_secs_f64() * 1e3,
    );

    // 2-4× target per the task spec. Soft floor at 1.5× so the test
    // fails loudly if parallelism regresses to a wash, but doesn't
    // false-positive on a 2-core CI runner.
    assert!(
        speedup4 >= 1.5,
        "threads=4 speedup {speedup4:.2}× below 1.5× floor (t1={t1:?}, t4={t4:?})"
    );
}
