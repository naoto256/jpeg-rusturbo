//! Decode-side EXIF / ICC pass-through tests. Mirror of
//! `tests/metadata.rs` (which covers the encode-side wire emission).
//!
//! Strategy: encode → decode roundtrip via the public API. The
//! encoder path is already known-good (see `metadata.rs` for wire
//! framing checks); here we verify the new `Decoder::exif()` /
//! `Decoder::icc_profile()` accessors return what the encoder put in.
//! Plus a handful of malformed / out-of-order / unrelated-APPn cases
//! built by hand-crafting raw JPEG byte streams.

use jpeg_rusturbo::{JpegEncoder, PixelFormat, decode::Decoder};

const W: u32 = 16;
const H: u32 = 16;

fn solid_rgb(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        out.extend_from_slice(&rgb);
    }
    out
}

fn encode(exif: Option<Vec<u8>>, icc: Option<Vec<u8>>, w: u32, h: u32) -> Vec<u8> {
    let pixels = solid_rgb(w, h, [128, 64, 200]);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_exif(exif);
        enc.set_icc_profile(icc);
        enc.encode_rgb(&pixels, w, h).unwrap();
    }
    out
}

// ---- EXIF roundtrip ----

#[test]
fn exif_roundtrip_small() {
    let exif: Vec<u8> = b"II\x2A\x00\x08\x00\x00\x00\x00\x00\x00\x00".to_vec();
    let jpeg = encode(Some(exif.clone()), None, W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(dec.exif(), Some(&exif[..]));
    assert!(dec.icc_profile().is_none());
}

#[test]
fn exif_roundtrip_medium() {
    let exif: Vec<u8> = (0..50u32).map(|i| (i & 0xFF) as u8).collect();
    let jpeg = encode(Some(exif.clone()), None, W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(dec.exif(), Some(&exif[..]));
}

#[test]
fn exif_roundtrip_large() {
    // 60 KB EXIF — still single APP1 (cap is ~65527 bytes after Exif\0\0).
    let exif: Vec<u8> = (0..60_000u32)
        .map(|i| (i.wrapping_mul(7) & 0xFF) as u8)
        .collect();
    let jpeg = encode(Some(exif.clone()), None, W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(dec.exif(), Some(&exif[..]));
}

#[test]
fn exif_empty_payload() {
    let exif: Vec<u8> = Vec::new();
    let jpeg = encode(Some(exif.clone()), None, W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    // An empty EXIF payload still has the Exif\0\0 identifier, so
    // exif() reports Some(&[]).
    assert_eq!(dec.exif(), Some(&exif[..]));
}

// ---- ICC roundtrip ----

#[test]
fn icc_roundtrip_small() {
    let icc: Vec<u8> = (0..1024u32).map(|i| (i & 0xFF) as u8).collect();
    let jpeg = encode(None, Some(icc.clone()), W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(dec.icc_profile(), Some(&icc[..]));
    assert!(dec.exif().is_none());
}

#[test]
fn icc_roundtrip_multi_segment() {
    // 150 KB → 3 segments at the 65519-byte chunk cap.
    let icc: Vec<u8> = (0..150_000u32)
        .map(|i| (i.wrapping_mul(31) & 0xFF) as u8)
        .collect();
    let jpeg = encode(None, Some(icc.clone()), W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    let got = dec.icc_profile().expect("ICC missing");
    assert_eq!(got.len(), icc.len(), "reassembled length mismatch");
    assert_eq!(got, &icc[..], "reassembled bytes don't match input");
}

#[test]
fn icc_roundtrip_chunk_boundary() {
    // Exactly the single-segment cap (65519 bytes) — still one APP2.
    let icc: Vec<u8> = (0..65_519u32)
        .map(|i| (i.wrapping_mul(13) & 0xFF) as u8)
        .collect();
    let jpeg = encode(None, Some(icc.clone()), W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(dec.icc_profile(), Some(&icc[..]));
}

#[test]
fn icc_cache_idempotent() {
    // Second call returns the same slice as the first (= cached
    // OnceCell, not a re-assembly).
    let icc: Vec<u8> = (0..1024u32).map(|i| (i & 0xFF) as u8).collect();
    let jpeg = encode(None, Some(icc.clone()), W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    let a = dec.icc_profile().unwrap().as_ptr();
    let b = dec.icc_profile().unwrap().as_ptr();
    assert_eq!(a, b, "ICC accessor should return the cached slice");
}

// ---- Combined ----

#[test]
fn exif_plus_icc() {
    let exif: Vec<u8> = b"II\x2A\x00\x08\x00\x00\x00\x00\x00\x00\x00".to_vec();
    let icc: Vec<u8> = (0..2048u32).map(|i| (i & 0xFF) as u8).collect();
    let jpeg = encode(Some(exif.clone()), Some(icc.clone()), W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(dec.exif(), Some(&exif[..]));
    assert_eq!(dec.icc_profile(), Some(&icc[..]));
}

// ---- No metadata ----

#[test]
fn no_metadata_returns_none() {
    let jpeg = encode(None, None, W, H);
    let dec = Decoder::new(&jpeg).unwrap();
    assert!(dec.exif().is_none());
    assert!(dec.icc_profile().is_none());
}

// ---- Pixel decode unaffected ----

#[test]
fn decode_pixels_with_metadata_works() {
    let exif: Vec<u8> = b"II\x2A\x00\x08\x00\x00\x00\x00\x00\x00\x00".to_vec();
    let icc: Vec<u8> = (0..1024u32).map(|i| (i & 0xFF) as u8).collect();
    let jpeg = encode(Some(exif), Some(icc), 32, 32);
    let dec = Decoder::new(&jpeg).unwrap();
    let info = dec.info();
    assert_eq!(info.width, 32);
    assert_eq!(info.height, 32);
    let pixels = dec.decode(PixelFormat::Rgb).unwrap();
    assert_eq!(pixels.len(), 32 * 32 * 3);
}

// ---- Cross-decoder: image crate sees same ICC bytes ----

#[test]
fn image_crate_icc_agrees() {
    use image::ImageDecoder;
    let icc: Vec<u8> = (0..4096u32)
        .map(|i| (i.wrapping_mul(17) & 0xFF) as u8)
        .collect();
    let jpeg = encode(None, Some(icc.clone()), 64, 64);
    // image::codecs::jpeg::JpegDecoder exposes icc_profile() via the
    // ImageDecoder trait since image 0.25.
    let mut img_dec =
        image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&jpeg)).expect("image decode");
    let icc_via_image = img_dec
        .icc_profile()
        .expect("icc_profile io")
        .expect("icc present");
    assert_eq!(icc_via_image, icc, "image crate ICC differs from ours");
    let our_dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(our_dec.icc_profile().unwrap(), &icc[..]);
    // EXIF: image's JpegDecoder doesn't expose APP1 EXIF on the
    // ImageDecoder trait surface (only ICC is standardized there);
    // skipped intentionally.
}

// ---- Hand-crafted malformed / out-of-order cases ----
//
// These build a minimal valid JPEG by encoding a tiny image, then
// rewrite the APP1 / APP2 prefix region. The entropy data after SOS
// is preserved verbatim so pixel decode still works.

/// Build a marker-only prefix replacement: takes a JPEG, locates the
/// first SOS marker (0xFF 0xDA), and returns
/// `(soi_and_jfif_only_prefix, sos_onwards_suffix)`. The caller
/// constructs the metadata region in between.
fn split_at_sof(jpeg: &[u8]) -> (Vec<u8>, Vec<u8>) {
    // Walk segments from SOI, copying SOI + JFIF (APP0) into prefix,
    // then everything from the first non-APP0 segment onwards into
    // suffix.
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    let mut prefix = vec![0xFF, 0xD8];
    let mut i = 2usize;
    // Expect APP0 immediately.
    assert_eq!(jpeg[i], 0xFF);
    let marker = jpeg[i + 1];
    assert_eq!(marker, 0xE0, "expected APP0 right after SOI");
    let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
    prefix.extend_from_slice(&jpeg[i..i + 2 + len]);
    i += 2 + len;
    // Skip any further APP1/APP2 segments the encoder wrote (we'll
    // synthesize our own); the suffix starts at the next non-APPn
    // segment (typically DQT 0xFF 0xDB).
    while i + 4 <= jpeg.len() {
        let m = jpeg[i + 1];
        if (0xE0..=0xEF).contains(&m) {
            let l = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            i += 2 + l;
        } else {
            break;
        }
    }
    let suffix = jpeg[i..].to_vec();
    (prefix, suffix)
}

fn app1_exif_segment(payload: &[u8]) -> Vec<u8> {
    let mut s = vec![0xFF, 0xE1];
    // Length field counts itself (2 bytes) + Exif\0\0 ID (6) + payload.
    let len_field = (2 + 6 + payload.len()) as u16;
    s.extend_from_slice(&len_field.to_be_bytes());
    s.extend_from_slice(b"Exif\0\0");
    s.extend_from_slice(payload);
    s
}

fn app2_icc_segment(seq: u8, total: u8, chunk: &[u8]) -> Vec<u8> {
    let mut s = vec![0xFF, 0xE2];
    let len_field = (2 + 12 + 2 + chunk.len()) as u16;
    s.extend_from_slice(&len_field.to_be_bytes());
    s.extend_from_slice(b"ICC_PROFILE\0");
    s.push(seq);
    s.push(total);
    s.extend_from_slice(chunk);
    s
}

fn app1_xmp_segment(payload: &[u8]) -> Vec<u8> {
    let mut s = vec![0xFF, 0xE1];
    const XMP_ID: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
    let len_field = (2 + XMP_ID.len() + payload.len()) as u16;
    s.extend_from_slice(&len_field.to_be_bytes());
    s.extend_from_slice(XMP_ID);
    s.extend_from_slice(payload);
    s
}

fn app3_unrelated_segment(payload: &[u8]) -> Vec<u8> {
    let mut s = vec![0xFF, 0xE3];
    let len_field = (2 + payload.len()) as u16;
    s.extend_from_slice(&len_field.to_be_bytes());
    s.extend_from_slice(payload);
    s
}

#[test]
fn icc_out_of_order_seq_reassembles() {
    let base = encode(None, None, 32, 32);
    let (prefix, suffix) = split_at_sof(&base);
    let chunk_size = 100;
    let icc: Vec<u8> = (0..3 * chunk_size).map(|i| (i & 0xFF) as u8).collect();
    let mut jpeg = prefix;
    // Emit seg 2 first, then seg 1, then seg 3.
    jpeg.extend(app2_icc_segment(2, 3, &icc[chunk_size..2 * chunk_size]));
    jpeg.extend(app2_icc_segment(1, 3, &icc[..chunk_size]));
    jpeg.extend(app2_icc_segment(3, 3, &icc[2 * chunk_size..]));
    jpeg.extend(suffix);
    let dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(dec.icc_profile(), Some(&icc[..]));
}

#[test]
fn icc_missing_seq_returns_none() {
    let base = encode(None, None, 32, 32);
    let (prefix, suffix) = split_at_sof(&base);
    let chunk: Vec<u8> = (0..100u8).collect();
    let mut jpeg = prefix;
    // total=3, emit only seq 1 and 3 — gap at seq 2.
    jpeg.extend(app2_icc_segment(1, 3, &chunk));
    jpeg.extend(app2_icc_segment(3, 3, &chunk));
    jpeg.extend(suffix);
    let dec = Decoder::new(&jpeg).unwrap();
    assert!(dec.icc_profile().is_none(), "missing seq should be None");
    // Pixel decode still works.
    let _ = dec.decode(PixelFormat::Rgb).unwrap();
}

#[test]
fn icc_duplicate_seq_first_wins() {
    let base = encode(None, None, 32, 32);
    let (prefix, suffix) = split_at_sof(&base);
    let chunk_a: Vec<u8> = vec![0xAA; 64];
    let chunk_b: Vec<u8> = vec![0xBB; 64];
    let chunk_2: Vec<u8> = vec![0x22; 64];
    let mut jpeg = prefix;
    jpeg.extend(app2_icc_segment(1, 2, &chunk_a));
    jpeg.extend(app2_icc_segment(1, 2, &chunk_b)); // duplicate of seq 1
    jpeg.extend(app2_icc_segment(2, 2, &chunk_2));
    jpeg.extend(suffix);
    let dec = Decoder::new(&jpeg).unwrap();
    let mut expected = chunk_a.clone();
    expected.extend_from_slice(&chunk_2);
    assert_eq!(dec.icc_profile(), Some(&expected[..]));
}

#[test]
fn xmp_app1_is_not_exif() {
    let base = encode(None, None, 32, 32);
    let (prefix, suffix) = split_at_sof(&base);
    let xmp_payload = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>";
    let mut jpeg = prefix;
    jpeg.extend(app1_xmp_segment(xmp_payload));
    jpeg.extend(suffix);
    let dec = Decoder::new(&jpeg).unwrap();
    assert!(
        dec.exif().is_none(),
        "XMP APP1 must not be reported as EXIF"
    );
    // And pixel decode still works.
    let _ = dec.decode(PixelFormat::Rgb).unwrap();
}

#[test]
fn unrelated_appn_segments_ignored() {
    let base = encode(None, None, 32, 32);
    let (prefix, suffix) = split_at_sof(&base);
    let exif: Vec<u8> = b"II\x2A\x00\x08\x00\x00\x00\x00\x00\x00\x00".to_vec();
    let mut jpeg = prefix;
    jpeg.extend(app3_unrelated_segment(b"hi from APP3"));
    jpeg.extend(app1_exif_segment(&exif));
    // APP14 Adobe-like (0xEE) — just give it a short body.
    let mut app14 = vec![0xFF, 0xEE];
    app14.extend_from_slice(&[0x00, 0x05, 0xAA, 0xBB, 0xCC]); // length=5, 3 body bytes
    jpeg.extend(app14);
    jpeg.extend(suffix);
    let dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(dec.exif(), Some(&exif[..]));
    assert!(dec.icc_profile().is_none());
    let _ = dec.decode(PixelFormat::Rgb).unwrap();
}

#[test]
fn multiple_exif_first_wins() {
    let base = encode(None, None, 32, 32);
    let (prefix, suffix) = split_at_sof(&base);
    let exif_a: Vec<u8> = b"II\x2A\x00first-exif".to_vec();
    let exif_b: Vec<u8> = b"II\x2A\x00second-exif".to_vec();
    let mut jpeg = prefix;
    jpeg.extend(app1_exif_segment(&exif_a));
    jpeg.extend(app1_exif_segment(&exif_b));
    jpeg.extend(suffix);
    let dec = Decoder::new(&jpeg).unwrap();
    assert_eq!(dec.exif(), Some(&exif_a[..]));
}
