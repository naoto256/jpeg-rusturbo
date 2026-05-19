//! JPEG marker stream reader (inverse of `crate::markers`).
//!
//! Walks the byte stream from SOI to SOS extracting headers (DQT,
//! DHT, SOF, SOS, DRI); the SOS step hands control over to the
//! entropy decoder. Standalone markers (SOI, EOI, RSTn) carry no
//! length field; everything else has a 2-byte big-endian length
//! covering the length bytes themselves.
//!
//! References: ITU-T T.81 Annex B, libjpeg `jdmarker.c`.

use super::error::{DecodeError, Result};

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
#[allow(dead_code)] // ss/se/ah/al used by progressive (0.3.0 roadmap)
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
    fn read_marker(&mut self) -> Result<u8> {
        // Skip fill 0xFF bytes (T.81 B.1.1.2 allows any number of them).
        let first = self.read_u8()?;
        if first != 0xFF {
            return Err(DecodeError::Malformed("expected 0xFF marker prefix"));
        }
        loop {
            let m = self.read_u8()?;
            if m != 0xFF {
                if m == 0x00 {
                    return Err(DecodeError::Malformed("stray 0xFF 0x00 outside scan"));
                }
                return Ok(m);
            }
        }
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
                        },
                        scan,
                    ));
                }
                M_EOI => return Err(DecodeError::Malformed("EOI before SOS")),
                M_COM | 0xE0..=0xEF => {
                    // Comment / APPn — skip the payload.
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
