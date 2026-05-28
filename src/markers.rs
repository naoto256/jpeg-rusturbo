//! JPEG marker / segment writers.
//!
//! All markers are byte-exact per ITU-T T.81 (the same a libjpeg
//! decoder produces). This module is pure I/O — no arithmetic, no
//! algorithm choices — so the SIMD kernels don't touch it.

use std::io::{self, Write};

use crate::tables::{StdHuffman, ZIGZAG};

/// Write a 16-bit big-endian word to the output.
fn write_be_u16<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

/// SOI — Start Of Image. (B.2.4.1)
pub fn write_soi<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_all(&[0xFF, 0xD8])
}

/// EOI — End Of Image. (B.2.4.2)
pub fn write_eoi<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_all(&[0xFF, 0xD9])
}

/// APP0 / JFIF identifier segment. Conventional 16-byte version 1.01
/// payload with no thumbnail. (B.5)
pub fn write_app0_jfif<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_all(&[0xFF, 0xE0])?;
    write_be_u16(w, 16)?; // segment length (excludes marker, includes itself)
    w.write_all(b"JFIF\0")?; // identifier
    w.write_all(&[1, 1])?; // version 1.01
    w.write_all(&[0])?; // units = 0 (no aspect ratio)
    write_be_u16(w, 1)?; // X density
    write_be_u16(w, 1)?; // Y density
    w.write_all(&[0, 0])?; // X/Y thumbnail
    Ok(())
}

/// Maximum payload bytes in a single APPn segment. JPEG segment length
/// is a 16-bit field counting itself + payload, so payload ≤ 65533.
const APP_PAYLOAD_MAX: usize = 65533;

/// APP1 / EXIF segment. The standard EXIF identifier `"Exif\0\0"` is
/// prepended; the caller supplies the raw EXIF bytes (typically a TIFF
/// header followed by IFD entries).
///
/// EXIF is permitted to occupy at most one APP1 segment, so the
/// payload limit is `APP_PAYLOAD_MAX - 6` (= 65527) bytes; oversize
/// inputs return `InvalidInput`. In practice EXIF blobs from cameras
/// are a few KB at most, so the limit is theoretical.
pub fn write_app1_exif<W: Write>(w: &mut W, exif: &[u8]) -> io::Result<()> {
    const ID: &[u8] = b"Exif\0\0";
    let payload_len = ID.len() + exif.len();
    if payload_len > APP_PAYLOAD_MAX {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "EXIF payload too large: {} bytes (max {} after Exif\\0\\0 prefix)",
                exif.len(),
                APP_PAYLOAD_MAX - ID.len(),
            ),
        ));
    }
    w.write_all(&[0xFF, 0xE1])?;
    write_be_u16(w, (2 + payload_len) as u16)?;
    w.write_all(ID)?;
    w.write_all(exif)
}

/// APP2 / ICC profile segment(s). The ICC.1 spec embeds profiles via
/// the identifier `"ICC_PROFILE\0"` followed by a 1-based
/// `(sequence_number, total_segments)` byte pair, allowing a profile
/// to span up to 255 APP2 segments. Small profiles (≤ 65519 bytes,
/// after subtracting the 14-byte header) fit in one segment.
///
/// Returns `InvalidInput` if the profile exceeds the addressable
/// `255 × max_chunk` capacity (~16.7 MB), which no realistic ICC
/// profile reaches.
pub fn write_app2_icc<W: Write>(w: &mut W, icc: &[u8]) -> io::Result<()> {
    const ID: &[u8] = b"ICC_PROFILE\0";
    // Per-segment payload after the 12-byte ID + 2-byte (seq, total).
    const CHUNK_MAX: usize = APP_PAYLOAD_MAX - 12 - 2;
    let total_segs = icc.len().div_ceil(CHUNK_MAX).max(1);
    if total_segs > 255 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "ICC profile too large: {} bytes needs {} segments (255 max per ICC.1)",
                icc.len(),
                total_segs,
            ),
        ));
    }
    for seg in 0..total_segs {
        let start = seg * CHUNK_MAX;
        let end = (start + CHUNK_MAX).min(icc.len());
        let chunk = &icc[start..end];
        let payload_len = ID.len() + 2 + chunk.len();
        w.write_all(&[0xFF, 0xE2])?;
        write_be_u16(w, (2 + payload_len) as u16)?;
        w.write_all(ID)?;
        // Sequence and total are 1-based per ICC.1 § B.4.
        w.write_all(&[(seg + 1) as u8, total_segs as u8])?;
        w.write_all(chunk)?;
    }
    Ok(())
}

/// DQT — Define Quantization Table. Writes one 8-bit precision table
/// at the supplied destination index `tq` (0 or 1). Coefficients are
/// emitted in zig-zag order. (B.2.4.1)
pub fn write_dqt<W: Write>(w: &mut W, tq: u8, table: &[u8; 64]) -> io::Result<()> {
    w.write_all(&[0xFF, 0xDB])?;
    write_be_u16(w, 67)?; // length: 2 + 1 + 64
    w.write_all(&[tq & 0x0F])?; // precision (0 = 8-bit) | dest id
    for &k in &ZIGZAG {
        w.write_all(&[table[k]])?;
    }
    Ok(())
}

/// SOF0 — Start Of Frame, baseline DCT. Encodes image dimensions and
/// per-component sampling factors / quant table assignments.
///
/// `components` items: (component id, h_samp, v_samp, quant_table_id).
/// For Y'CbCr we use ids 1, 2, 3.
pub fn write_sof0<W: Write>(
    w: &mut W,
    width: u16,
    height: u16,
    components: &[(u8, u8, u8, u8)],
) -> io::Result<()> {
    w.write_all(&[0xFF, 0xC0])?;
    let n = components.len() as u16;
    write_be_u16(w, 8 + 3 * n)?; // length
    w.write_all(&[8])?; // sample precision (8-bit)
    write_be_u16(w, height)?;
    write_be_u16(w, width)?;
    w.write_all(&[n as u8])?;
    for &(id, h, v, tq) in components {
        w.write_all(&[id, ((h & 0xF) << 4) | (v & 0xF), tq])?;
    }
    Ok(())
}

/// DHT — Define Huffman Table. Writes one table (DC or AC) at class
/// `tc` (0 = DC, 1 = AC), destination `th`. (B.2.4.2)
pub fn write_dht<W: Write>(w: &mut W, tc: u8, th: u8, table: &StdHuffman) -> io::Result<()> {
    write_dht_bits_values(w, tc, th, &table.bits, table.values)
}

/// DHT writer working from raw bits[16] + values[] (used by the
/// optimized-Huffman path, which builds its tables at runtime instead
/// of pointing at a `StdHuffman` constant).
pub fn write_dht_bits_values<W: Write>(
    w: &mut W,
    tc: u8,
    th: u8,
    bits: &[u8; 16],
    values: &[u8],
) -> io::Result<()> {
    let n_values = values.len() as u16;
    w.write_all(&[0xFF, 0xC4])?;
    write_be_u16(w, 2 + 1 + 16 + n_values)?;
    w.write_all(&[((tc & 0xF) << 4) | (th & 0xF)])?;
    w.write_all(bits)?;
    w.write_all(values)?;
    Ok(())
}

/// DRI — Define Restart Interval. Sets the number of MCUs between
/// RSTn markers in the entropy data. `interval == 0` disables restart
/// markers (this segment is then unnecessary; the encoder skips it).
pub fn write_dri<W: Write>(w: &mut W, interval: u16) -> io::Result<()> {
    w.write_all(&[0xFF, 0xDD])?;
    write_be_u16(w, 4)?; // length
    write_be_u16(w, interval)?;
    Ok(())
}

/// SOS — Start Of Scan. Identifies the components in this scan and
/// their DC/AC table assignments. We always write a single
/// interleaved scan over all components (baseline sequential).
///
/// `components` items: (component id, dc_tab_id, ac_tab_id).
pub fn write_sos<W: Write>(w: &mut W, components: &[(u8, u8, u8)]) -> io::Result<()> {
    // Baseline sequential: full spectral range 0..=63 with no
    // successive approximation.
    write_sos_spectral(w, components, 0, 63, 0, 0)
}

/// SOS for progressive-mode scans. Identical framing to
/// [`write_sos`], but the trailing 3 bytes carry the scan's spectral
/// selection (Ss..Se) and successive approximation (Ah / Al). See
/// T.81 G.2 — the same MCU-ordering rules apply, but each scan emits
/// only a sub-band or refines a previously-sent sub-band.
pub fn write_sos_spectral<W: Write>(
    w: &mut W,
    components: &[(u8, u8, u8)],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
) -> io::Result<()> {
    w.write_all(&[0xFF, 0xDA])?;
    let n = components.len() as u16;
    write_be_u16(w, 6 + 2 * n)?;
    w.write_all(&[n as u8])?;
    for &(id, dc, ac) in components {
        w.write_all(&[id, ((dc & 0xF) << 4) | (ac & 0xF)])?;
    }
    w.write_all(&[ss, se, ((ah & 0xF) << 4) | (al & 0xF)])?;
    Ok(())
}

/// SOF2 — Start Of Frame, progressive DCT. Same layout as
/// [`write_sof0`] except for the marker byte (0xC2 instead of 0xC0);
/// the decoder uses the marker to know whether to expect one scan
/// (sequential) or many (progressive).
pub fn write_sof2<W: Write>(
    w: &mut W,
    width: u16,
    height: u16,
    components: &[(u8, u8, u8, u8)],
) -> io::Result<()> {
    w.write_all(&[0xFF, 0xC2])?;
    let n = components.len() as u16;
    write_be_u16(w, 8 + 3 * n)?;
    w.write_all(&[8])?;
    write_be_u16(w, height)?;
    write_be_u16(w, width)?;
    w.write_all(&[n as u8])?;
    for &(id, h, v, tq) in components {
        w.write_all(&[id, ((h & 0xF) << 4) | (v & 0xF), tq])?;
    }
    Ok(())
}
