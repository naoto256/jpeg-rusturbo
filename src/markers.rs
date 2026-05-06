//! JPEG marker / segment writers.
//!
//! All markers are byte-exact per ITU-T T.81 (the same a libjpeg
//! decoder produces). This module is pure I/O — no arithmetic, no
//! algorithm choices — so Phase 2's NEON work doesn't touch it.

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
    write_be_u16(w, 16)?;            // segment length (excludes marker, includes itself)
    w.write_all(b"JFIF\0")?;          // identifier
    w.write_all(&[1, 1])?;            // version 1.01
    w.write_all(&[0])?;               // units = 0 (no aspect ratio)
    write_be_u16(w, 1)?;             // X density
    write_be_u16(w, 1)?;             // Y density
    w.write_all(&[0, 0])?;            // X/Y thumbnail
    Ok(())
}

/// DQT — Define Quantization Table. Writes one 8-bit precision table
/// at the supplied destination index `tq` (0 or 1). Coefficients are
/// emitted in zig-zag order. (B.2.4.1)
pub fn write_dqt<W: Write>(w: &mut W, tq: u8, table: &[u8; 64]) -> io::Result<()> {
    w.write_all(&[0xFF, 0xDB])?;
    write_be_u16(w, 67)?;            // length: 2 + 1 + 64
    w.write_all(&[tq & 0x0F])?;       // precision (0 = 8-bit) | dest id
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
    write_be_u16(w, 8 + 3 * n)?;     // length
    w.write_all(&[8])?;               // sample precision (8-bit)
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
pub fn write_dht<W: Write>(
    w: &mut W,
    tc: u8,
    th: u8,
    table: &StdHuffman,
) -> io::Result<()> {
    let n_values = table.values.len() as u16;
    w.write_all(&[0xFF, 0xC4])?;
    write_be_u16(w, 2 + 1 + 16 + n_values)?;
    w.write_all(&[((tc & 0xF) << 4) | (th & 0xF)])?;
    w.write_all(&table.bits)?;
    w.write_all(table.values)?;
    Ok(())
}

/// SOS — Start Of Scan. Identifies the components in this scan and
/// their DC/AC table assignments. We always write a single
/// interleaved scan over all components (baseline sequential).
///
/// `components` items: (component id, dc_tab_id, ac_tab_id).
pub fn write_sos<W: Write>(
    w: &mut W,
    components: &[(u8, u8, u8)],
) -> io::Result<()> {
    w.write_all(&[0xFF, 0xDA])?;
    let n = components.len() as u16;
    write_be_u16(w, 6 + 2 * n)?;
    w.write_all(&[n as u8])?;
    for &(id, dc, ac) in components {
        w.write_all(&[id, ((dc & 0xF) << 4) | (ac & 0xF)])?;
    }
    // Spectral selection start/end + successive approximation: fixed
    // values for baseline sequential.
    w.write_all(&[0, 63, 0])?;
    Ok(())
}
