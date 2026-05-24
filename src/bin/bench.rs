//! Unified micro-benchmark for jpeg-rusturbo's encode pipeline.
//!
//! Build with `cargo run --release --bin bench -- [--section <name>]`.
//! The build label reflects the active arch backend selected at compile time
//! (use `--features force-scalar` to disable SIMD kernels in the same binary).
//!
//! Sections:
//!   A — encode pipeline      : 3 resolutions × 3 subsampling × q=80, default knobs
//!   B — threads scaling      : 2 resolutions × threads {1,2,4,8,auto} × q=80, 4:2:0
//!   C — optimize-huffman size: 3 resolutions × 3 subsampling × q {70,80,90} × {off,on}
//!   D — decode pipeline      : 3 resolutions × 3 subsampling × q=80 (decode of A's output)
//!   all (default)            : run A then B then C then D
//!
//! Decode timing vs the `image` crate (cross-crate comparison) lives in
//! `tests/comparison_bench.rs` (run with
//! `cargo test --release --test comparison_bench -- --ignored --nocapture`).

use std::env;
use std::time::Instant;

use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder, PixelFormat, decode};

const WARMUP: usize = 3;
const ITERATIONS: usize = 50;

const RES_ALL: &[(&str, u32, u32)] = &[
    ("1592x1124 (session-size)", 1592, 1124),
    ("1920x1080 (1080p)", 1920, 1080),
    ("3840x2160 (4K)", 3840, 2160),
];
const RES_THREADS: &[(&str, u32, u32)] = &[
    ("1920x1080 (1080p)", 1920, 1080),
    ("3840x2160 (4K)", 3840, 2160),
];

const SUBSAMP_ALL: &[(&str, ChromaSubsampling)] = &[
    ("4:4:4", ChromaSubsampling::Yuv444),
    ("4:2:2", ChromaSubsampling::Yuv422),
    ("4:2:0", ChromaSubsampling::Yuv420),
];

fn main() {
    let section = parse_section();
    print_header(section);

    match section {
        Section::A => bench_a(),
        Section::B => bench_b(),
        Section::C => bench_c(),
        Section::D => bench_d(),
        Section::All => {
            bench_a();
            bench_b();
            bench_c();
            bench_d();
        }
    }
}

#[derive(Copy, Clone)]
enum Section {
    A,
    B,
    C,
    D,
    All,
}

fn parse_section() -> Section {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--section" {
            let name = args.next().unwrap_or_else(|| usage_and_exit());
            return match name.as_str() {
                "A" | "a" => Section::A,
                "B" | "b" => Section::B,
                "C" | "c" => Section::C,
                "D" | "d" => Section::D,
                "all" | "All" => Section::All,
                _ => usage_and_exit(),
            };
        } else if arg == "--help" || arg == "-h" {
            usage_and_exit();
        }
    }
    Section::All
}

fn usage_and_exit() -> ! {
    eprintln!("usage: bench [--section A|B|C|D|all]");
    std::process::exit(2);
}

fn print_header(section: Section) {
    let arch = if cfg!(all(target_arch = "aarch64", not(feature = "force-scalar"))) {
        "aarch64 (NEON: main kernels + huffman bitmap)"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64 (force-scalar — NEON kernels disabled)"
    } else if cfg!(all(target_arch = "x86_64", not(feature = "force-scalar"))) {
        "x86_64 (AVX2: main kernels; SSE2: huffman bitmap)"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64 (force-scalar — AVX2 kernels disabled)"
    } else {
        "scalar (other arch)"
    };
    let profile = if cfg!(debug_assertions) {
        "debug (numbers will be useless)"
    } else {
        "release"
    };
    let sec = match section {
        Section::A => "A (encode pipeline)",
        Section::B => "B (threads scaling)",
        Section::C => "C (optimize-huffman size)",
        Section::D => "D (decode pipeline)",
        Section::All => "all",
    };
    println!("jpeg-rusturbo bench — section: {sec}");
    println!("build: {arch}, profile: {profile}");
    println!("warmup: {WARMUP} iter, measured: {ITERATIONS} iter");
}

// ---------------------------------------------------------------------------
// Section A — encode pipeline, defaults
// ---------------------------------------------------------------------------

fn bench_a() {
    println!("\n=== A. encode pipeline (q=80, threads=1, optimize-huffman=off) ===");
    for &(sub_label, sub) in SUBSAMP_ALL {
        println!("\n  subsampling: {sub_label}");
        for &(label, w, h) in RES_ALL {
            let pixels = make_image(w, h);
            let r = time_encode(&pixels, w, h, |enc| {
                enc.set_subsampling(sub);
            });
            print_row(label, w, h, &r);
        }
    }
}

// ---------------------------------------------------------------------------
// Section B — threads scaling
// ---------------------------------------------------------------------------

fn bench_b() {
    println!("\n=== B. threads scaling (q=80, 4:2:0, optimize-huffman=off) ===");
    let thread_settings: &[(&str, u32)] = &[
        ("threads=1", 1),
        ("threads=2", 2),
        ("threads=4", 4),
        ("threads=8", 8),
        ("threads=auto", 0),
    ];
    for &(label, w, h) in RES_THREADS {
        println!("\n  {label}");
        for &(t_label, t) in thread_settings {
            let pixels = make_image(w, h);
            let r = time_encode(&pixels, w, h, |enc| {
                enc.set_subsampling(ChromaSubsampling::Yuv420);
                enc.set_threads(t);
            });
            print_row(t_label, w, h, &r);
        }
    }
}

// ---------------------------------------------------------------------------
// Section C — optimize-huffman size delta
// ---------------------------------------------------------------------------

fn bench_c() {
    println!("\n=== C. optimize-huffman size delta (threads=1) ===");
    for &(sub_label, sub) in SUBSAMP_ALL {
        for &q in &[70u8, 80, 90] {
            println!("\n  subsampling: {sub_label}, q={q}");
            for &(label, w, h) in RES_ALL {
                let pixels = make_image(w, h);
                let off = time_encode_q(&pixels, w, h, q, |enc| {
                    enc.set_subsampling(sub);
                });
                let on = time_encode_q(&pixels, w, h, q, |enc| {
                    enc.set_subsampling(sub);
                    enc.set_optimize_huffman(true);
                });
                let delta_pct = (on.size as f64 - off.size as f64) / off.size as f64 * 100.0;
                println!(
                    "    {label:<28}  off: {off_size:>9} B {off_ms:>7.2} ms  |  on: {on_size:>9} B {on_ms:>7.2} ms  Δ {delta:>+6.2}%",
                    off_size = off.size,
                    off_ms = off.ms_per_iter,
                    on_size = on.size,
                    on_ms = on.ms_per_iter,
                    delta = delta_pct,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Section D — decode pipeline
// ---------------------------------------------------------------------------

fn bench_d() {
    println!("\n=== D. decode pipeline (q=80, JPEG -> RGB, default knobs) ===");
    for &(sub_label, sub) in SUBSAMP_ALL {
        println!("\n  subsampling: {sub_label}");
        for &(label, w, h) in RES_ALL {
            let pixels = make_image(w, h);

            // Encode once outside the timed loop; the JPEG bytes are
            // deterministic, so we can reuse them across all iterations.
            let mut jpeg = Vec::with_capacity((w as usize * h as usize * 3) / 2);
            {
                let mut enc = JpegEncoder::new_with_quality(&mut jpeg, 80);
                enc.set_subsampling(sub);
                enc.encode_rgba(&pixels, w, h).unwrap();
            }

            let r = time_decode(&jpeg, w, h);
            let mp = (w as f64) * (h as f64) / 1_000_000.0;
            let ms_per_mp = r.ms_per_iter / mp;
            println!(
                "    {label:<28}  {ms:>7.2} ms/iter   {ms_per_mp:>5.2} ms/MPx   (JPEG {size} B → RGB {rgb_size} B)",
                ms = r.ms_per_iter,
                size = jpeg.len(),
                rgb_size = r.size,
            );
        }
    }
}

fn time_decode(jpeg: &[u8], w: u32, h: u32) -> Result {
    let rgb_size = (w as usize) * (h as usize) * 3;
    let mut rgb = Vec::with_capacity(rgb_size);

    // Warm-up.
    for _ in 0..WARMUP {
        rgb = decode::decode(jpeg, PixelFormat::Rgb).unwrap();
        std::hint::black_box(&rgb);
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        rgb = decode::decode(jpeg, PixelFormat::Rgb).unwrap();
        std::hint::black_box(&rgb);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / ITERATIONS as u32;
    let mp = (w as f64) * (h as f64) / 1_000_000.0;
    Result {
        ms_per_iter: per_iter.as_secs_f64() * 1000.0,
        ms_per_mp: per_iter.as_secs_f64() * 1000.0 / mp,
        size: rgb.len(),
    }
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

struct Result {
    ms_per_iter: f64,
    ms_per_mp: f64,
    size: usize,
}

fn time_encode<F: Fn(&mut JpegEncoder<&mut Vec<u8>>)>(
    pixels: &[u8],
    w: u32,
    h: u32,
    configure: F,
) -> Result {
    time_encode_q(pixels, w, h, 80, configure)
}

fn time_encode_q<F: Fn(&mut JpegEncoder<&mut Vec<u8>>)>(
    pixels: &[u8],
    w: u32,
    h: u32,
    quality: u8,
    configure: F,
) -> Result {
    let mut buf = Vec::with_capacity((w as usize * h as usize * 3) / 2);

    for _ in 0..WARMUP {
        buf.clear();
        let mut enc = JpegEncoder::new_with_quality(&mut buf, quality);
        configure(&mut enc);
        enc.encode_rgba(pixels, w, h).unwrap();
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        buf.clear();
        let mut enc = JpegEncoder::new_with_quality(&mut buf, quality);
        configure(&mut enc);
        enc.encode_rgba(pixels, w, h).unwrap();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / ITERATIONS as u32;
    let mp = (w as f64) * (h as f64) / 1_000_000.0;
    Result {
        ms_per_iter: per_iter.as_secs_f64() * 1000.0,
        ms_per_mp: per_iter.as_secs_f64() * 1000.0 / mp,
        size: buf.len(),
    }
}

fn print_row(label: &str, _w: u32, _h: u32, r: &Result) {
    println!(
        "    {label:<28}  {ms:>7.2} ms/iter   {mp:>5.2} ms/MPx   ({size} bytes)",
        ms = r.ms_per_iter,
        mp = r.ms_per_mp,
        size = r.size,
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
