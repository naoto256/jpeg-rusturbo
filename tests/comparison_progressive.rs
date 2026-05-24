//! Read-direction comparison tests: decode each vendored JPEG with
//! our decoder and the `image` crate, assert per-channel max diff
//! and PSNR floors. Covers baseline + progressive in 4:4:4 / 4:2:0
//! with edge sizes (4x4, 17x17, 32x32, 650x470) and grayscale.
//!
//! Corpus is in `tests/fixtures/progressive/`. Provenance: vendored
//! from `image-rs/jpeg-decoder`'s reftest suite (MIT OR Apache-2.0)
//! — see `NOTICE.md`. The corpus is intentionally small and
//! independent of any single competing decoder so the comparison is
//! "we agree with `image` on JPEGs neither of us authored".
//!
//! These fixtures *won't* be shipped to crates.io (`exclude =
//! ["tests/"]` in `Cargo.toml`).

use image::{ImageFormat, ImageReader};
use jpeg_rusturbo::PixelFormat;
use jpeg_rusturbo::decode::Decoder;
use std::io::Cursor;

/// Threshold per fixture, exposed so individual cases can opt for a
/// looser bound when the spec semantics permit (e.g. when our
/// decoder's clamping rule legitimately diverges by 1 from `image`'s).
#[derive(Clone, Copy)]
struct Thresholds {
    /// Maximum per-channel absolute difference vs. `image`'s output.
    max_diff: i32,
    /// Minimum PSNR in dB vs. `image`'s output. `f64::INFINITY` to
    /// require exact byte-identical decode.
    min_psnr: f64,
}

const DEFAULT: Thresholds = Thresholds {
    max_diff: 3,
    min_psnr: 40.0,
};

fn psnr(a: &[u8], b: &[u8]) -> f64 {
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

fn assert_decode_agrees(path: &str, thresholds: Thresholds) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));

    // Our decoder.
    let dec = Decoder::new(&bytes).unwrap_or_else(|e| panic!("our headers {path}: {e:?}"));
    let info = dec.info();
    let ours = dec
        .decode(PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("our decode {path}: {e:?}"));

    // `image` as ground truth (note: `image` uses zune-jpeg under the
    // hood for the actual decode; this is a "two independent
    // implementations agree" check, not an absolute spec audit).
    let theirs = ImageReader::with_format(Cursor::new(&bytes), ImageFormat::Jpeg)
        .decode()
        .unwrap_or_else(|e| panic!("image decode {path}: {e:?}"))
        .to_rgb8()
        .into_raw();

    assert_eq!(
        ours.len(),
        theirs.len(),
        "{path}: output length mismatch ours={} theirs={} ({}x{} {}c)",
        ours.len(),
        theirs.len(),
        info.width,
        info.height,
        info.components,
    );

    let mut max_diff = 0i32;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        let d = (*a as i32 - *b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    let p = psnr(&ours, &theirs);
    assert!(
        max_diff <= thresholds.max_diff,
        "{path}: max per-channel diff {max_diff} > floor {} ({}x{} {}c prog={})",
        thresholds.max_diff,
        info.width,
        info.height,
        info.components,
        info.progressive,
    );
    assert!(
        p >= thresholds.min_psnr,
        "{path}: PSNR {:.2} dB below floor {:.2} dB",
        p,
        thresholds.min_psnr,
    );
}

#[test]
fn baseline_grayscale_32x32() {
    assert_decode_agrees(
        "tests/fixtures/progressive/jpg-gray.jpg",
        Thresholds {
            max_diff: 1,
            min_psnr: 50.0,
        },
    );
}

#[test]
fn baseline_420_odd_size_17x17() {
    assert_decode_agrees(
        "tests/fixtures/progressive/jpg-size-17x17.jpg",
        Thresholds {
            max_diff: 3,
            min_psnr: 40.0,
        },
    );
}

/// `partial_progressive.jpg` carries 18 bytes of stray padding between
/// the last DHT and the final SOS — typical of encoders that leave
/// trailing junk in intermediate progressive output. Both image and
/// our decoder skip the padding via tolerant `read_marker`. At 4x4 the
/// fancy-upsample edge handling diverges by ≤ 8/channel from image's
/// (16 pixels total, every one of them is an edge); the looser bound
/// just acknowledges that, the key assertion is "we decode it at all".
#[test]
fn progressive_444_partial_4x4_with_stray_bytes() {
    assert_decode_agrees(
        "tests/fixtures/progressive/partial_progressive.jpg",
        Thresholds {
            max_diff: 8,
            min_psnr: 30.0,
        },
    );
}

#[test]
fn progressive_444_medium_650x470() {
    assert_decode_agrees("tests/fixtures/progressive/progressive3.jpg", DEFAULT);
}

#[test]
fn progressive_420_32x32() {
    assert_decode_agrees("tests/fixtures/progressive/jpg-progressive.jpg", DEFAULT);
}
