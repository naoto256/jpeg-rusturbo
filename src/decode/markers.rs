//! JPEG marker stream reader (inverse of `crate::encode::markers`).
//!
//! Walks the byte stream from SOI to SOS extracting headers (DQT,
//! DHT, SOF, SOS, DRI); the SOS step hands control over to the
//! entropy decoder. Standalone markers (SOI, EOI, RSTn) carry no
//! length field; everything else has a 2-byte big-endian length
//! covering the length bytes themselves.
//!
//! References: ITU-T T.81 Annex B, libjpeg `jdmarker.c`.

use super::error::{DecodeError, Result};

use std::collections::BTreeMap;
use std::ops::Range;

// ---- Marker code constants (ITU-T T.81 B.1.1) ----

pub const M_SOI: u8 = 0xD8;
pub const M_EOI: u8 = 0xD9;
pub const M_SOS: u8 = 0xDA;
pub const M_DQT: u8 = 0xDB;
pub const M_DHT: u8 = 0xC4;
pub const M_DRI: u8 = 0xDD;
pub const M_SOF0: u8 = 0xC0; // baseline DCT Huffman
pub const M_SOF2: u8 = 0xC2; // progressive DCT Huffman
pub const M_COM: u8 = 0xFE;
// APPn = 0xE0..=0xEF
// RSTn = 0xD0..=0xD7 (in-scan only)

/// Per-component descriptor parsed from SOF.
#[derive(Clone, Copy, Debug)]
pub struct Component {
    /// Component identifier (Ci). 1=Y, 2=Cb, 3=Cr by convention.
    pub id: u8,
    /// Horizontal sampling factor (Hi). 1, 2, or 4.
    pub h: u8,
    /// Vertical sampling factor (Vi).
    pub v: u8,
    /// Quantization table selector (Tqi), 0..=3.
    pub qt: u8,
}

/// Frame header parsed from SOF0 / SOF2.
#[derive(Clone, Debug)]
#[allow(dead_code)] // `precision` carried for future 12-bit support
pub struct FrameHeader {
    /// Sample precision in bits (P). We accept 8 only.
    pub precision: u8,
    /// Image height (Y), in samples.
    pub height: u16,
    /// Image width (X), in samples.
    pub width: u16,
    /// Components in declaration order (usually Y, Cb, Cr for 3-component).
    pub components: Vec<Component>,
    /// True for SOF2 progressive, false for SOF0 baseline.
    pub progressive: bool,
}

impl FrameHeader {
    /// Max horizontal sampling factor across components (`Hmax`).
    pub fn h_max(&self) -> u8 {
        self.components.iter().map(|c| c.h).max().unwrap_or(1)
    }
    /// Max vertical sampling factor across components (`Vmax`).
    pub fn v_max(&self) -> u8 {
        self.components.iter().map(|c| c.v).max().unwrap_or(1)
    }
    /// MCU geometry in pixels: `(h_max * 8, v_max * 8)`.
    #[allow(dead_code)] // used by progressive scan and downstream consumers
    pub fn mcu_pixels(&self) -> (u32, u32) {
        ((self.h_max() as u32) * 8, (self.v_max() as u32) * 8)
    }
}

/// Per-component scan parameters from SOS.
#[derive(Clone, Copy, Debug)]
pub struct ScanComponent {
    /// Which frame component this entry refers to (Csj).
    pub component_id: u8,
    /// DC Huffman table selector (Tdj), 0..=3.
    pub dc_table: u8,
    /// AC Huffman table selector (Taj), 0..=3.
    pub ac_table: u8,
}

/// Scan header parsed from SOS.
#[derive(Clone, Debug)]
pub struct ScanHeader {
    pub components: Vec<ScanComponent>,
    /// Start of spectral selection (Ss). 0 for DC, 1..=63 for AC.
    pub ss: u8,
    /// End of spectral selection (Se). 63 for baseline / DC scans.
    pub se: u8,
    /// Successive approximation high (Ah). 0 for baseline / first scans.
    pub ah: u8,
    /// Successive approximation low (Al). 0 for baseline.
    pub al: u8,
}

/// Quantization table — natural order, 8-bit precision only for now.
#[derive(Clone, Copy, Debug)]
pub struct QuantTable {
    /// Destination identifier (Tq), 0..=3.
    pub id: u8,
    /// Natural-order entries (the DQT segment carries them in zig-zag
    /// order; this struct stores them already de-zig-zagged).
    pub values: [u16; 64],
}

/// Huffman table — raw `bits` + `values` payload, as in DHT.
/// Table-building (canonical-Huffman expansion to a decode LUT)
/// happens in `super::huffman`.
#[derive(Clone, Debug)]
pub struct HuffmanTableSpec {
    /// Table class (Tc): 0 = DC, 1 = AC.
    pub class: u8,
    /// Destination identifier (Th), 0..=3.
    pub id: u8,
    /// `bits[i]` = number of codes of length `i+1`, i in 0..16.
    pub bits: [u8; 16],
    /// Symbols, in code order. Length = sum of `bits`.
    pub values: Vec<u8>,
}

/// Output of the pre-scan marker walk: everything the decoder needs
/// before starting the first scan.
#[derive(Clone, Debug)]
pub struct DecoderHeaders {
    pub frame: FrameHeader,
    pub quant: Vec<QuantTable>,
    pub huffman: Vec<HuffmanTableSpec>,
    pub restart_interval: u16,
    pub metadata: DecoderMetadata,
}

/// Pass-through metadata collected from APP1 / APP2 segments during
/// the marker walk. Bytes are stored as ranges into the original JPEG
/// source buffer so EXIF retrieval is zero-copy; ICC needs reassembly
/// of one or more chunks at access time.
///
/// Per the EXIF spec there is at most one APP1 `Exif\0\0` segment per
/// file; if multiple are present the first one wins. Other APP1 flavours
/// (XMP "http://ns.adobe.com/xap/1.0/\0" etc.) are intentionally ignored
/// here — the bytes are still consumed without error, but `exif` stays
/// `None`. Surfacing XMP would need a separate accessor.
#[derive(Clone, Debug, Default)]
pub struct DecoderMetadata {
    /// Range into the source buffer pointing at the EXIF payload AFTER
    /// the 6-byte `Exif\0\0` identifier.
    pub exif: Option<Range<usize>>,
    /// `seq_num` (1-based) → range into the source buffer pointing at
    /// the per-segment chunk AFTER the 14-byte `ICC_PROFILE\0` +
    /// `seq` + `total` header. Duplicate `seq_num` values: first wins.
    pub icc_chunks: BTreeMap<u8, Range<usize>>,
    /// Value of the `total_segments` byte from the first ICC segment
    /// seen. Subsequent segments with a different `total` are ignored
    /// (treated as malformed mid-stream metadata).
    pub icc_total: Option<u8>,
}

/// Cursor over the JPEG byte stream.
pub struct MarkerReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> MarkerReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Resume parsing at byte offset `pos`. Used by the progressive
    /// decoder: after a scan finishes the bit-reader leaves the cursor
    /// immediately past the marker that terminated entropy data, and we
    /// need a fresh `MarkerReader` to walk any intervening DHT/DQT/DRI
    /// segments before the next SOS.
    pub fn resume_at(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    /// Current byte position, useful when handing off to the entropy
    /// decoder mid-stream.
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[allow(dead_code)] // useful for diagnostic dumps; kept on the type
    pub fn remaining(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    fn read_u8(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let hi = self.read_u8()? as u16;
        let lo = self.read_u8()? as u16;
        Ok((hi << 8) | lo)
    }

    fn read_slice(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(DecodeError::Malformed("length overflow"))?;
        if end > self.buf.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// Read the next marker code: skip any 0xFF fill bytes (T.81 B.1.1.2),
    /// then expect a non-0x00, non-0xFF byte as the marker id.
    ///
    /// Tolerant variant: stray bytes between segments (some encoders leave
    /// trailing padding or junk between SOS / DHT / EOI; see e.g.
    /// `partial_progressive.jpg` in our test corpus) are silently skipped
    /// until the next `0xFF` is found. libjpeg-turbo behaves the same
    /// way; without this we'd reject streams the rest of the ecosystem
    /// reads fine.
    fn read_marker(&mut self) -> Result<u8> {
        // Skip non-0xFF garbage until the next marker prefix or EOF.
        while self.pos < self.buf.len() && self.buf[self.pos] != 0xFF {
            self.pos += 1;
        }
        if self.pos >= self.buf.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        // Consume the 0xFF and any 0xFF fill bytes that follow.
        while self.pos < self.buf.len() && self.buf[self.pos] == 0xFF {
            self.pos += 1;
        }
        if self.pos >= self.buf.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let m = self.buf[self.pos];
        self.pos += 1;
        if m == 0x00 {
            return Err(DecodeError::Malformed("stray 0xFF 0x00 outside scan"));
        }
        Ok(m)
    }

    /// Read header markers up to and including SOS, returning the
    /// aggregated header set. On return, `self.pos()` points at the
    /// first byte of entropy-coded data.
    pub fn read_to_scan(&mut self) -> Result<(DecoderHeaders, ScanHeader)> {
        // Expect SOI first.
        if self.read_u8()? != 0xFF || self.read_u8()? != M_SOI {
            return Err(DecodeError::Malformed("missing SOI marker"));
        }

        let mut frame: Option<FrameHeader> = None;
        let mut quant: Vec<QuantTable> = Vec::new();
        let mut huffman: Vec<HuffmanTableSpec> = Vec::new();
        let mut restart_interval: u16 = 0;
        let mut metadata = DecoderMetadata::default();

        loop {
            let marker = self.read_marker()?;
            match marker {
                M_SOF0 | M_SOF2 => {
                    let frame_hdr = self.parse_sof(marker == M_SOF2)?;
                    frame = Some(frame_hdr);
                }
                M_DQT => self.parse_dqt(&mut quant)?,
                M_DHT => self.parse_dht(&mut huffman)?,
                M_DRI => {
                    let len = self.read_u16()?;
                    if len != 4 {
                        return Err(DecodeError::Malformed("bad DRI length"));
                    }
                    restart_interval = self.read_u16()?;
                }
                M_SOS => {
                    let frame = frame.ok_or(DecodeError::Malformed("SOS before SOF"))?;
                    let scan = self.parse_sos(&frame)?;
                    return Ok((
                        DecoderHeaders {
                            frame,
                            quant,
                            huffman,
                            restart_interval,
                            metadata,
                        },
                        scan,
                    ));
                }
                M_EOI => return Err(DecodeError::Malformed("EOI before SOS")),
                0xE1 => self.parse_app1(&mut metadata)?,
                0xE2 => self.parse_app2(&mut metadata)?,
                M_COM | 0xE0 | 0xE3..=0xEF => {
                    // Comment / APP0 (JFIF, handled elsewhere) / other
                    // APPn (Adobe XMP / Photoshop IRB / etc.) — skip
                    // the payload. APP1 EXIF and APP2 ICC are retained
                    // above; other APP1 / APP2 flavours fall through
                    // their own handlers as a no-op skip.
                    let len = self.read_u16()?;
                    if len < 2 {
                        return Err(DecodeError::Malformed("bad segment length"));
                    }
                    self.read_slice(len as usize - 2)?;
                }
                0xC1 | 0xC3..=0xC7 | 0xC9..=0xCF => {
                    return Err(DecodeError::Unsupported("non-baseline/progressive SOFn"));
                }
                _ => {
                    // Unknown segment — best effort skip if length-prefixed.
                    let len = self.read_u16()?;
                    if len < 2 {
                        return Err(DecodeError::Malformed("bad unknown segment length"));
                    }
                    self.read_slice(len as usize - 2)?;
                }
            }
        }
    }

    /// Walk markers between two progressive scans, returning either the
    /// next [`ScanHeader`] (SOS reached) or `None` (EOI reached). Updates
    /// any intervening DHT / DQT / DRI segments in-place so the caller
    /// can rebuild its Huffman / quantization tables before the next
    /// scan runs.
    ///
    /// `pending_marker` lets the caller pass in a marker that was
    /// already consumed by the entropy decoder (the BitReader pulls
    /// `0xFF <id>` off the byte stream when it spots a non-RST marker,
    /// so the marker prefix is gone before this reader picks up).
    pub fn next_scan_or_end(
        &mut self,
        frame: &FrameHeader,
        huffman: &mut Vec<HuffmanTableSpec>,
        quant: &mut Vec<QuantTable>,
        restart_interval: &mut u16,
        pending_marker: Option<u8>,
    ) -> Result<Option<ScanHeader>> {
        let mut next = pending_marker;
        loop {
            let marker = match next.take() {
                Some(m) => m,
                None => self.read_marker()?,
            };
            match marker {
                M_SOS => return Ok(Some(self.parse_sos(frame)?)),
                M_EOI => return Ok(None),
                M_DHT => self.parse_dht(huffman)?,
                M_DQT => self.parse_dqt(quant)?,
                M_DRI => {
                    let len = self.read_u16()?;
                    if len != 4 {
                        return Err(DecodeError::Malformed("bad DRI length"));
                    }
                    *restart_interval = self.read_u16()?;
                }
                M_COM | 0xE0..=0xEF => {
                    let len = self.read_u16()?;
                    if len < 2 {
                        return Err(DecodeError::Malformed("bad segment length"));
                    }
                    self.read_slice(len as usize - 2)?;
                }
                M_SOF0 | M_SOF2 | 0xC1 | 0xC3..=0xC7 | 0xC9..=0xCF => {
                    return Err(DecodeError::Malformed("unexpected SOF between scans"));
                }
                _ => {
                    let len = self.read_u16()?;
                    if len < 2 {
                        return Err(DecodeError::Malformed("bad unknown segment length"));
                    }
                    self.read_slice(len as usize - 2)?;
                }
            }
        }
    }

    /// APP1 segment dispatcher. Retains the payload as EXIF if the
    /// 6-byte `Exif\0\0` identifier prefix is present; otherwise (XMP,
    /// other Adobe namespaces) consumes the bytes without recording.
    /// At most one EXIF segment is retained — first-wins per the spec.
    fn parse_app1(&mut self, metadata: &mut DecoderMetadata) -> Result<()> {
        const ID: &[u8] = b"Exif\0\0";
        let len = self.read_u16()? as usize;
        if len < 2 {
            return Err(DecodeError::Malformed("bad APP1 length"));
        }
        let payload_len = len - 2;
        let payload_start = self.pos;
        let payload = self.read_slice(payload_len)?;
        if metadata.exif.is_none() && payload.len() >= ID.len() && payload.starts_with(ID) {
            metadata.exif = Some(payload_start + ID.len()..payload_start + payload.len());
        }
        Ok(())
    }

    /// APP2 segment dispatcher. Retains the chunk as part of an ICC
    /// profile if the 12-byte `ICC_PROFILE\0` identifier + (seq, total)
    /// header is present; otherwise consumes the bytes without
    /// recording. Multi-segment ICC profiles arrive in one APP2 per
    /// segment; assembly is deferred to access time in `Decoder`.
    fn parse_app2(&mut self, metadata: &mut DecoderMetadata) -> Result<()> {
        const ID: &[u8] = b"ICC_PROFILE\0";
        // ID (12) + seq (1) + total (1) = 14
        const HEADER_LEN: usize = 14;
        let len = self.read_u16()? as usize;
        if len < 2 {
            return Err(DecodeError::Malformed("bad APP2 length"));
        }
        let payload_len = len - 2;
        let payload_start = self.pos;
        let payload = self.read_slice(payload_len)?;
        if payload.len() < HEADER_LEN || !payload.starts_with(ID) {
            return Ok(());
        }
        let seq = payload[12];
        let total = payload[13];
        // Record the per-segment chunk range AFTER the 14-byte header.
        let chunk_start = payload_start + HEADER_LEN;
        let chunk_end = payload_start + payload.len();
        match metadata.icc_total {
            None => metadata.icc_total = Some(total),
            Some(prev) if prev != total => {
                // Inconsistent `total` across segments: mid-stream
                // disagreement is malformed. Leave the data captured so
                // far in place; `icc_profile()` will detect the gap /
                // mismatch at access time and return `None`. Skip
                // recording this segment so a single bad sender can't
                // overwrite a valid chunk.
                return Ok(());
            }
            _ => {}
        }
        // First-wins on duplicate seq.
        metadata
            .icc_chunks
            .entry(seq)
            .or_insert(chunk_start..chunk_end);
        Ok(())
    }

    fn parse_sof(&mut self, progressive: bool) -> Result<FrameHeader> {
        let len = self.read_u16()? as usize;
        if len < 8 {
            return Err(DecodeError::Malformed("SOF length too small"));
        }
        let precision = self.read_u8()?;
        if precision != 8 {
            return Err(DecodeError::Unsupported("non-8-bit precision"));
        }
        let height = self.read_u16()?;
        let width = self.read_u16()?;
        // Reject zero dimensions and oversized images. JPEG's wire format
        // allows up to 65535x65535, but blindly accepting that lets a
        // 50-byte malformed header demand multi-GB allocations downstream
        // (each component plane is `stride * padded_height` u8). The cap
        // here matches what mainstream Rust JPEG decoders enforce
        // (`image` / `jpeg-decoder` both gate around 16k).
        if width == 0 || height == 0 {
            return Err(DecodeError::InvalidDimensions("zero dimension"));
        }
        const MAX_DIMENSION: u16 = 16384;
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(DecodeError::InvalidDimensions(
                "image dimension exceeds 16384 (raise the cap explicitly to opt in)",
            ));
        }
        let nf = self.read_u8()? as usize;
        if len != 8 + 3 * nf {
            return Err(DecodeError::Malformed("SOF length / Nf mismatch"));
        }
        if nf == 0 || nf > 4 {
            return Err(DecodeError::Unsupported("unsupported component count"));
        }
        let mut components = Vec::with_capacity(nf);
        for _ in 0..nf {
            let id = self.read_u8()?;
            let hv = self.read_u8()?;
            let qt = self.read_u8()?;
            let h = hv >> 4;
            let v = hv & 0x0F;
            if !matches!(h, 1 | 2 | 4) || !matches!(v, 1 | 2 | 4) {
                return Err(DecodeError::Unsupported("sampling factor not 1/2/4"));
            }
            if qt > 3 {
                return Err(DecodeError::Malformed("quant table selector out of range"));
            }
            components.push(Component { id, h, v, qt });
        }
        // T.81 A.2.2 — for single-component scans the MCU is exactly
        // one data unit regardless of the declared H/V sampling factors.
        // Encoders that produce single-component frames with H=V=2 (some
        // grayscale tools do this for legacy reasons) would otherwise
        // confuse downstream block-grid sizing. Normalize to H=V=1 so
        // the per-component grid matches the decoded block count.
        if components.len() == 1 {
            components[0].h = 1;
            components[0].v = 1;
        }

        Ok(FrameHeader {
            precision,
            height,
            width,
            components,
            progressive,
        })
    }

    fn parse_dqt(&mut self, out: &mut Vec<QuantTable>) -> Result<()> {
        let len = self.read_u16()? as usize;
        if len < 2 {
            return Err(DecodeError::Malformed("DQT length too small"));
        }
        let mut remaining = len - 2;
        while remaining > 0 {
            let pq_tq = self.read_u8()?;
            remaining -= 1;
            let pq = pq_tq >> 4;
            let tq = pq_tq & 0x0F;
            if tq > 3 {
                return Err(DecodeError::Malformed("DQT id out of range"));
            }
            if pq != 0 {
                return Err(DecodeError::Unsupported("16-bit DQT precision"));
            }
            // 8-bit precision: 64 bytes.
            if remaining < 64 {
                return Err(DecodeError::Malformed("DQT short payload"));
            }
            let zz = self.read_slice(64)?;
            remaining -= 64;
            // De-zig-zag into natural order using crate::tables::ZIGZAG.
            let mut values = [0u16; 64];
            for (k, &v) in zz.iter().enumerate() {
                values[crate::tables::ZIGZAG[k]] = v as u16;
            }
            out.push(QuantTable { id: tq, values });
        }
        Ok(())
    }

    fn parse_dht(&mut self, out: &mut Vec<HuffmanTableSpec>) -> Result<()> {
        let len = self.read_u16()? as usize;
        if len < 2 {
            return Err(DecodeError::Malformed("DHT length too small"));
        }
        let mut remaining = len - 2;
        while remaining > 0 {
            let tc_th = self.read_u8()?;
            remaining -= 1;
            let tc = tc_th >> 4;
            let th = tc_th & 0x0F;
            if tc > 1 {
                return Err(DecodeError::Malformed("DHT class out of range"));
            }
            if th > 3 {
                return Err(DecodeError::Malformed("DHT id out of range"));
            }
            if remaining < 16 {
                return Err(DecodeError::Malformed("DHT bits truncated"));
            }
            let bits_slice = self.read_slice(16)?;
            remaining -= 16;
            let mut bits = [0u8; 16];
            bits.copy_from_slice(bits_slice);
            let total: usize = bits.iter().map(|&b| b as usize).sum();
            if total > 256 {
                return Err(DecodeError::Malformed("DHT symbol count > 256"));
            }
            if remaining < total {
                return Err(DecodeError::Malformed("DHT values truncated"));
            }
            let values_slice = self.read_slice(total)?;
            remaining -= total;
            out.push(HuffmanTableSpec {
                class: tc,
                id: th,
                bits,
                values: values_slice.to_vec(),
            });
        }
        Ok(())
    }

    fn parse_sos(&mut self, frame: &FrameHeader) -> Result<ScanHeader> {
        let len = self.read_u16()? as usize;
        if len < 6 {
            return Err(DecodeError::Malformed("SOS length too small"));
        }
        let ns = self.read_u8()? as usize;
        if ns == 0 || ns > 4 {
            return Err(DecodeError::Malformed("SOS Ns out of range"));
        }
        if len != 6 + 2 * ns {
            return Err(DecodeError::Malformed("SOS length / Ns mismatch"));
        }
        let mut components = Vec::with_capacity(ns);
        for _ in 0..ns {
            let cs = self.read_u8()?;
            let td_ta = self.read_u8()?;
            // Validate Cs maps to a known frame component.
            if !frame.components.iter().any(|c| c.id == cs) {
                return Err(DecodeError::Malformed("SOS component id not in frame"));
            }
            components.push(ScanComponent {
                component_id: cs,
                dc_table: td_ta >> 4,
                ac_table: td_ta & 0x0F,
            });
        }
        let ss = self.read_u8()?;
        let se = self.read_u8()?;
        let ah_al = self.read_u8()?;
        let ah = ah_al >> 4;
        let al = ah_al & 0x0F;
        Ok(ScanHeader {
            components,
            ss,
            se,
            ah,
            al,
        })
    }
}
