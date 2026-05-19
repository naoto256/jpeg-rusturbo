//! Decode-side bit reader + Huffman lookup.
//!
//! The bit reader handles 0xFF 0x00 byte-unstuffing and surfaces
//! non-RST markers so the scan loop knows when entropy data ends.
//! `HuffmanDecodeTable` is the canonical-Huffman → symbol lookup
//! built once per DHT, with an 8-bit fast LUT and a slow walk for
//! codes 9..16 bits long (libjpeg `jdhuff.c` style).
//!
//! References: ITU-T T.81 Annex C/F, libjpeg-turbo `jdhuff.c`.

use super::error::{DecodeError, Result};
use super::markers::HuffmanTableSpec;

/// MSB-first bit reader over the entropy-coded segment.
///
/// On encountering 0xFF 0x00, drops the 0x00 (byte stuffing).
/// On 0xFF followed by anything else, records the marker code and
/// pads further reads with zero bits (T.81 F.2.2.5).
pub struct BitReader<'a> {
    src: &'a [u8],
    pos: usize,
    buf: u64,
    nbits: u32,
    marker: Option<u8>,
}

impl<'a> BitReader<'a> {
    /// Construct a reader starting at byte `pos` in `src`. Typically
    /// `pos` is the offset returned by `MarkerReader` right after SOS.
    pub fn new(src: &'a [u8], pos: usize) -> Self {
        Self {
            src,
            pos,
            buf: 0,
            nbits: 0,
            marker: None,
        }
    }

    #[allow(dead_code)] // used by progressive scan path (on the 0.4.0 roadmap)
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// The marker code that terminated entropy data, if any (set when
    /// the reader encounters `0xFF <non-0x00>`). Once set, the reader
    /// is "drained": further `get_bits` calls return zero-padded
    /// values, and the scan loop should stop and dispatch on the
    /// marker.
    pub fn marker(&self) -> Option<u8> {
        self.marker
    }

    /// Discard whatever partial byte is currently buffered. Used after
    /// RST handling to realign on the next byte boundary.
    pub fn reset_bit_buffer(&mut self) {
        self.buf = 0;
        self.nbits = 0;
    }

    /// Clear a previously-seen marker so the reader continues with the
    /// next byte. Used by RSTn handling.
    pub fn clear_marker(&mut self) {
        self.marker = None;
    }

    fn fill(&mut self, need: u32) -> Result<()> {
        debug_assert!(need <= 32);
        while self.nbits < need {
            if self.marker.is_some() {
                self.buf <<= 8;
                self.nbits += 8;
                continue;
            }
            let b = *self.src.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
            self.pos += 1;
            if b == 0xFF {
                let next = *self.src.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
                self.pos += 1;
                if next == 0x00 {
                    self.buf = (self.buf << 8) | 0xFF;
                    self.nbits += 8;
                } else {
                    self.marker = Some(next);
                    self.buf <<= 8;
                    self.nbits += 8;
                }
            } else {
                self.buf = (self.buf << 8) | (b as u64);
                self.nbits += 8;
            }
        }
        Ok(())
    }

    /// Peek the next `n` bits without consuming them. `n` must be in
    /// 1..=16. Caller must ensure `fill` was called first.
    fn peek_bits(&self, n: u32) -> u32 {
        debug_assert!((1..=16).contains(&n));
        debug_assert!(n <= self.nbits);
        ((self.buf >> (self.nbits - n)) as u32) & ((1u32 << n) - 1)
    }

    fn drop_bits(&mut self, n: u32) {
        debug_assert!(n <= self.nbits);
        self.nbits -= n;
        if self.nbits == 0 {
            self.buf = 0;
        } else {
            self.buf &= (1u64 << self.nbits) - 1;
        }
    }

    /// Read `n` bits, MSB-first. `n` in 1..=16.
    pub fn get_bits(&mut self, n: u32) -> Result<u32> {
        if self.nbits < n {
            self.fill(n)?;
        }
        let v = self.peek_bits(n);
        self.drop_bits(n);
        Ok(v)
    }

    /// Read 1 bit. Used by progressive AC refine and 1-bit refinement.
    #[allow(dead_code)] // used by progressive scan path (on the 0.4.0 roadmap)
    pub fn get_bit(&mut self) -> Result<u32> {
        self.get_bits(1)
    }

    /// Decode one Huffman symbol using `tbl`.
    pub fn decode_symbol(&mut self, tbl: &HuffmanDecodeTable) -> Result<u8> {
        // Fast path: peek 8 bits, look up.
        if self.nbits < 8 {
            self.fill(8)?;
        }
        let idx = self.peek_bits(8) as usize;
        let entry = tbl.look[idx];
        if entry != 0 {
            let length = (entry >> 8) as u32;
            let symbol = (entry & 0xFF) as u8;
            self.drop_bits(length);
            return Ok(symbol);
        }
        // Slow path: drop 8 bits and walk lengths 9..16.
        self.drop_bits(8);
        let mut code = idx as i32;
        for l in 9..=16 {
            if self.nbits < 1 {
                self.fill(1)?;
            }
            code = (code << 1) | (self.peek_bits(1) as i32);
            self.drop_bits(1);
            if code <= tbl.max_code[l] {
                let huff_index = (code + tbl.val_offset[l]) as usize;
                if huff_index >= tbl.values.len() {
                    return Err(DecodeError::Malformed("huffman index out of range"));
                }
                return Ok(tbl.values[huff_index]);
            }
        }
        Err(DecodeError::Malformed("invalid huffman code (>16 bits)"))
    }
}

/// Decode-side Huffman lookup table.
pub struct HuffmanDecodeTable {
    /// 8-bit fast LUT. `look[next_8_bits] = (length << 8) | symbol`,
    /// with `length == 0` indicating "no symbol fits in 8 bits — fall
    /// to slow path".
    look: [u16; 256],
    /// `max_code[l]` = largest huff code of length `l` (l in 1..=16),
    /// or `i32::MIN` if no codes of that length. Index 0 unused.
    max_code: [i32; 17],
    /// `val_offset[l]` such that `huff_value_index = code + val_offset[l]`
    /// for a code of length `l`.
    val_offset: [i32; 17],
    /// Symbol table from DHT (HUFFVAL).
    values: Vec<u8>,
}

impl HuffmanDecodeTable {
    pub fn from_spec(spec: &HuffmanTableSpec) -> Result<Self> {
        // 1. Build huffsize[]: per-symbol code length.
        let mut huffsize: Vec<u8> = Vec::with_capacity(spec.values.len());
        for (l_idx, &count) in spec.bits.iter().enumerate() {
            for _ in 0..count {
                huffsize.push((l_idx + 1) as u8);
            }
        }
        if huffsize.len() != spec.values.len() {
            return Err(DecodeError::Malformed("DHT bits/values mismatch"));
        }
        if huffsize.is_empty() {
            return Err(DecodeError::Malformed("empty DHT table"));
        }

        // 2. Build huffcode[]: canonical Huffman code per symbol.
        let mut huffcode: Vec<u32> = Vec::with_capacity(huffsize.len());
        let mut code: u32 = 0;
        let mut si = huffsize[0] as u32;
        let mut p = 0usize;
        loop {
            while p < huffsize.len() && (huffsize[p] as u32) == si {
                huffcode.push(code);
                code = code.wrapping_add(1);
                p += 1;
            }
            if p == huffsize.len() {
                break;
            }
            // Bump length until we match the next group's size.
            while (huffsize[p] as u32) != si {
                code <<= 1;
                si += 1;
            }
        }

        // 3. Build the 8-bit fast LUT.
        let mut look = [0u16; 256];
        let mut p_fast = 0usize;
        for l in 1..=8usize {
            for _ in 0..spec.bits[l - 1] {
                let huff_code = huffcode[p_fast];
                let huff_val = spec.values[p_fast];
                let shifted = (huff_code << (8 - l)) as usize;
                let count = 1usize << (8 - l);
                let entry = ((l as u16) << 8) | (huff_val as u16);
                for i in 0..count {
                    look[shifted + i] = entry;
                }
                p_fast += 1;
            }
        }

        // 4. Build max_code / val_offset for the slow path (l = 9..=16).
        // We populate l=1..16 in case a degenerate table has no
        // codes ≤ 8 bits, even though baseline JPEG won't.
        let mut max_code = [i32::MIN; 17];
        let mut val_offset = [0i32; 17];
        let mut p_slow = 0usize;
        for l in 1..=16usize {
            let cnt = spec.bits[l - 1] as usize;
            if cnt > 0 {
                val_offset[l] = (p_slow as i32) - (huffcode[p_slow] as i32);
                p_slow += cnt;
                max_code[l] = huffcode[p_slow - 1] as i32;
            }
        }

        Ok(Self {
            look,
            max_code,
            val_offset,
            values: spec.values.clone(),
        })
    }
}

/// Convert a magnitude-category encoded value back to its signed
/// integer. Inverse of `magnitude_category` in the encoder
/// (`crate::huffman::magnitude_category`).
///
/// Given the size category `s` (1..=16) and the `s`-bit `bits` payload:
/// - If the top bit of `bits` is 1, the original value was positive,
///   and `bits` is the low `s` bits of the value.
/// - Otherwise, the original was negative, and `value = bits - (1 << s) + 1`.
#[inline]
pub fn extend(bits: u32, s: u32) -> i32 {
    debug_assert!((1..=16).contains(&s));
    let v = bits as i32;
    let half = 1i32 << (s - 1);
    if v < half { v + (-1i32 << s) + 1 } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extend_inverse_of_encoder_categories() {
        // Mirror cases from the encoder's tests.
        // 5 → size=3, bits=0b101 → extend(0b101, 3) = 5
        assert_eq!(extend(0b101, 3), 5);
        // -5 → size=3, bits=0b010 → extend(0b010, 3) = -5
        assert_eq!(extend(0b010, 3), -5);
        // 1 → size=1, bits=1 → extend(1, 1) = 1
        assert_eq!(extend(1, 1), 1);
        // -1 → size=1, bits=0 → extend(0, 1) = -1
        assert_eq!(extend(0, 1), -1);
        // -1023 → size=10, bits = 1's complement = (-1023 - 1) & ((1<<10)-1) = -1024 & 1023 = 0
        assert_eq!(extend(0, 10), -1023);
        // 1023 → size=10, bits=1023 → extend(1023, 10) = 1023
        assert_eq!(extend(1023, 10), 1023);
    }

    #[test]
    fn bit_reader_basic() {
        let data = [0b1010_1100u8, 0b0011_0101];
        let mut br = BitReader::new(&data, 0);
        assert_eq!(br.get_bits(4).unwrap(), 0b1010);
        assert_eq!(br.get_bits(4).unwrap(), 0b1100);
        assert_eq!(br.get_bits(8).unwrap(), 0b0011_0101);
    }

    #[test]
    fn bit_reader_unstuffs_ff00() {
        // 0xFF 0x00 → 0xFF data byte.
        let data = [0xAB, 0xFF, 0x00, 0xCD];
        let mut br = BitReader::new(&data, 0);
        assert_eq!(br.get_bits(8).unwrap(), 0xAB);
        assert_eq!(br.get_bits(8).unwrap(), 0xFF);
        assert_eq!(br.get_bits(8).unwrap(), 0xCD);
    }

    #[test]
    fn bit_reader_surfaces_marker() {
        // 0xFF 0xD9 = EOI marker.
        let data = [0x12, 0xFF, 0xD9];
        let mut br = BitReader::new(&data, 0);
        assert_eq!(br.get_bits(8).unwrap(), 0x12);
        // Triggering a read advances past the marker; bits returned
        // afterward are zero-padded.
        let _ = br.get_bits(8).unwrap();
        assert_eq!(br.marker(), Some(0xD9));
    }

    #[test]
    fn huffman_table_decodes_standard_luma_dc() {
        // Build from the Annex K standard luma DC table.
        let spec = HuffmanTableSpec {
            class: 0,
            id: 0,
            bits: crate::tables::STD_LUMA_DC.bits,
            values: crate::tables::STD_LUMA_DC.values.to_vec(),
        };
        let tbl = HuffmanDecodeTable::from_spec(&spec).unwrap();
        // Symbol 0 has code 00 (length 2). Verify the LUT.
        // 00 left-shifted to fill 8 bits → 0x00..=0x3F all map to symbol 0
        // with length 2.
        for i in 0..0x40usize {
            let entry = tbl.look[i];
            assert_eq!(entry, 2u16 << 8, "look[{i:#x}]");
        }
    }
}
