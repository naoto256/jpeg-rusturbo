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
        // Fast refill: pull 4 bytes at once when no stuff byte or marker
        // is in the window. SWAR has-byte-equal-FF check (`(~x - 1) & x`
        // form, applied to the bitwise complement) lets the per-byte
        // loop handle only the rare 0xFF cases. Requires `nbits <= 32`
        // so the shifted-in chunk fits in `buf`.
        if self.marker.is_none() && self.nbits <= 32 && self.pos + 4 <= self.src.len() {
            let chunk = u32::from_be_bytes([
                self.src[self.pos],
                self.src[self.pos + 1],
                self.src[self.pos + 2],
                self.src[self.pos + 3],
            ]);
            let y = !chunk;
            let has_ff = y.wrapping_sub(0x0101_0101) & !y & 0x8080_8080;
            if has_ff == 0 {
                self.buf = (self.buf << 32) | (chunk as u64);
                self.nbits += 32;
                self.pos += 4;
                if self.nbits >= need {
                    return Ok(());
                }
            }
        }
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

/// T.81 C.2 canonical-Huffman expansion: from a DHT spec, return
/// `(huffsize, huffcode)` where `huffsize[i]` is the code length of
/// symbol `i` and `huffcode[i]` is its canonical code.
///
/// Shared by every Huffman LUT builder in this module
/// (`HuffmanDecodeTable`, `FastAcHuffmanTable`, `FastDcHuffmanTable`).
fn build_canonical_huffman(spec: &HuffmanTableSpec) -> Result<(Vec<u8>, Vec<u32>)> {
    // huffsize[]: per-symbol code length, by walking bits[].
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

    // huffcode[]: canonical Huffman code per symbol.
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

    Ok((huffsize, huffcode))
}

impl HuffmanDecodeTable {
    pub fn from_spec(spec: &HuffmanTableSpec) -> Result<Self> {
        // `huffsize` isn't needed here: HuffmanDecodeTable walks
        // `spec.bits[l-1]` directly for both the fast LUT (l=1..=8)
        // and the slow path tables (l=1..=16).
        let (_huffsize, huffcode) = build_canonical_huffman(spec)?;

        // Build the 8-bit fast LUT.
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

// ---------------------------------------------------------------
// Combined AC Huffman LUT (decode_symbol + get_bits in one lookup).
//
// JPEG baseline AC scan decodes each non-zero coefficient as:
//   1. decode_symbol(ac_tbl) → run/size byte
//   2. get_bits(size)        → magnitude bits
// Each step does its own peek/drop on the bit buffer. Combining the
// two into a single LUT keyed on the next PEEK_WIDTH bits lets the
// scan loop do one peek + one drop per coefficient when the symbol
// fits, and only fall back to the two-step path for long codes or
// codes whose magnitude bits spill past the peek window.
//
// The decode_block_baseline scan loop in this module's sibling
// `baseline.rs` calls `decode_ac_fast` first and falls back to the
// canonical `decode_symbol` + `get_bits` path only when the symbol
// misses (long code or magnitude bits spilling past the peek
// window). Bit-identity with the slow path is asserted by the
// cross-check tests below.
// ---------------------------------------------------------------

/// Width of the LUT key in bits. 10 bits → 1024 entries / 4 KiB.
/// Chosen so that almost all baseline AC symbols (code length 2..7
/// for the common run/size bytes) plus a small magnitude (0..3 bits)
/// land in the fast path; longer codes (length 11..16 in the standard
/// tables) and large-magnitude symbols correctly fall back.
pub const FAST_AC_PEEK_WIDTH: u32 = 10;

const FAST_AC_LUT_SIZE: usize = 1 << FAST_AC_PEEK_WIDTH;

/// One LUT entry. Packed u32, layout:
///   bits  0..7  : run (0..=15)
///   bits  8..15 : size (0..=15; baseline AC tops out at 10)
///   bits 16..23 : total_consumed = code_length + size (0..=PEEK_WIDTH)
///   bit   24    : valid (1 = fast path applies; 0 = fall back to slow path)
///   bits 25..31 : reserved (0)
#[derive(Clone, Copy)]
pub struct FastAcEntry(u32);

impl FastAcEntry {
    const INVALID: Self = Self(0);

    #[inline]
    fn new(run: u8, size: u8, total_consumed: u8) -> Self {
        Self((run as u32) | ((size as u32) << 8) | ((total_consumed as u32) << 16) | (1u32 << 24))
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        (self.0 >> 24) & 1 == 1
    }
    #[inline]
    pub fn run(self) -> u8 {
        self.0 as u8
    }
    #[inline]
    pub fn size(self) -> u8 {
        (self.0 >> 8) as u8
    }
    #[inline]
    pub fn total_consumed(self) -> u8 {
        (self.0 >> 16) as u8
    }
}

/// Combined AC lookup table indexed by the next `FAST_AC_PEEK_WIDTH`
/// bits of the entropy stream.
pub struct FastAcHuffmanTable {
    lut: Box<[FastAcEntry; FAST_AC_LUT_SIZE]>,
}

impl FastAcHuffmanTable {
    /// Build from a DHT spec (AC class). Uses the shared
    /// `build_canonical_huffman` helper to expand `(huffsize,
    /// huffcode)`; then for every symbol whose code_length + size ≤
    /// PEEK_WIDTH, fills the LUT slots prefixed by [code | every
    /// magnitude variant | every don't-care tail] with the same
    /// packed entry.
    pub fn from_spec(spec: &HuffmanTableSpec) -> Result<Self> {
        let (huffsize, huffcode) = build_canonical_huffman(spec)?;

        // Fill the LUT. Codes whose length itself exceeds
        // PEEK_WIDTH, and codes where code_length + size exceeds
        // PEEK_WIDTH, are left at valid=0 — the canonical Huffman
        // prefix property guarantees their LUT slots don't collide
        // with any fast-path symbol.
        let mut lut = Box::new([FastAcEntry::INVALID; FAST_AC_LUT_SIZE]);
        for i in 0..huffsize.len() {
            let l = huffsize[i] as u32;
            if l > FAST_AC_PEEK_WIDTH {
                continue;
            }
            let sym = spec.values[i];
            let run = (sym >> 4) & 0x0F;
            let size = (sym & 0x0F) as u32;
            let total = l + size;
            if total > FAST_AC_PEEK_WIDTH {
                continue;
            }
            let entry = FastAcEntry::new(run, size as u8, total as u8);
            // Slots prefixed by `code` (left-aligned to PEEK_WIDTH).
            // 2^(PEEK_WIDTH - l) slots in total, all of which carry
            // this same (run, size, total) — the magnitude bits and
            // the don't-care tail are extracted at lookup time.
            let base = (huffcode[i] as usize) << (FAST_AC_PEEK_WIDTH - l);
            let span = 1usize << (FAST_AC_PEEK_WIDTH - l);
            for j in 0..span {
                lut[base + j] = entry;
            }
        }

        Ok(Self { lut })
    }

    /// Return the entry for the given PEEK_WIDTH-bit peek key.
    #[inline]
    pub fn lookup(&self, peek_key: u32) -> FastAcEntry {
        self.lut[(peek_key & ((1u32 << FAST_AC_PEEK_WIDTH) - 1)) as usize]
    }

    /// Number of LUT slots flagged valid. Useful for diagnostics and
    /// for asserting that the fast path actually covers the bulk of
    /// the input distribution on the standard tables.
    #[allow(dead_code)] // diagnostic-only: consumed by the lut-report and coverage tests
    pub fn fast_path_slot_count(&self) -> usize {
        self.lut.iter().filter(|e| e.is_valid()).count()
    }
}

/// One coefficient decoded via the combined LUT.
///
/// `magnitude_raw` is the raw `size`-bit magnitude field as it
/// appears in the stream, before sign-extension. Callers apply
/// [`extend`] to recover the signed value.
pub struct FastAcDecoded {
    pub run: u8,
    pub size: u8,
    pub magnitude_raw: u32,
    /// Bits consumed from the stream on this hit. The scan loop doesn't
    /// read it (`decode_ac_fast` already advanced the reader), but it's
    /// kept for cross-checking against the slow path in tests.
    #[allow(dead_code)]
    pub total_consumed: u8,
}

/// Combined AC fast-path lookup.
///
/// Returns `Some(FastAcDecoded)` if the next `FAST_AC_PEEK_WIDTH`
/// bits hit a valid LUT entry — in that case the reader is advanced
/// by `total_consumed` bits and the caller can apply
/// `extend(magnitude_raw, size)` (when `size > 0`) to recover the
/// signed coefficient. Returns `Ok(None)` if the LUT slot is invalid;
/// the caller must then fall back to
/// `BitReader::decode_symbol` + `BitReader::get_bits` (and the bit
/// buffer is left untouched so the slow path sees the same bits).
pub fn decode_ac_fast(
    br: &mut BitReader,
    tbl: &FastAcHuffmanTable,
) -> Result<Option<FastAcDecoded>> {
    if br.nbits < FAST_AC_PEEK_WIDTH {
        br.fill(FAST_AC_PEEK_WIDTH)?;
    }
    let peek_key =
        ((br.buf >> (br.nbits - FAST_AC_PEEK_WIDTH)) as u32) & ((1u32 << FAST_AC_PEEK_WIDTH) - 1);
    let entry = tbl.lookup(peek_key);
    if !entry.is_valid() {
        return Ok(None);
    }
    let run = entry.run();
    let size = entry.size();
    let total = entry.total_consumed();
    let code_length = total - size;
    let magnitude_raw = if size == 0 {
        0
    } else {
        // Magnitude bits sit inside `peek_key` at offset code_length
        // from the MSB of the PEEK_WIDTH-bit window.
        let shift = FAST_AC_PEEK_WIDTH - code_length as u32 - size as u32;
        (peek_key >> shift) & ((1u32 << size) - 1)
    };
    br.drop_bits(total as u32);
    Ok(Some(FastAcDecoded {
        run,
        size,
        magnitude_raw,
        total_consumed: total,
    }))
}

// ---------------------------------------------------------------
// Combined DC Huffman LUT (decode_symbol + get_bits in one lookup).
//
// JPEG DC term per block is:
//   1. decode_symbol(dc_tbl) → size byte (magnitude bit count)
//   2. get_bits(size)        → magnitude bits
// Same pattern as the AC fast path above, but the symbol is just
// `size` (no run nibble) and total_consumed = code_length + size
// where size can be up to 11 in baseline (DC diffs in -2047..=2047).
// Per-block this only fires once vs. up to 63 AC terms, so the
// expected win is small (~0.5-1%); the change is mainly for
// completeness of the combined-LUT coverage of the scan loop.
// ---------------------------------------------------------------

/// Width of the DC LUT key in bits. Sized to match the AC LUT for
/// uniformity, even though DC symbols are typically shorter so the
/// fast path's hit rate is correspondingly higher.
pub const FAST_DC_PEEK_WIDTH: u32 = 10;

const FAST_DC_LUT_SIZE: usize = 1 << FAST_DC_PEEK_WIDTH;

/// One DC LUT entry. Packed u32, layout:
///   bits  0..7  : size (= magnitude bits count, 0..=11)
///   bits  8..15 : total_consumed = code_length + size (0..=PEEK_WIDTH)
///   bit  16     : valid (1 = fast path applies; 0 = fall back)
///   bits 17..31 : reserved (0)
#[derive(Clone, Copy)]
pub struct FastDcEntry(u32);

impl FastDcEntry {
    const INVALID: Self = Self(0);

    #[inline]
    fn new(size: u8, total_consumed: u8) -> Self {
        Self((size as u32) | ((total_consumed as u32) << 8) | (1u32 << 16))
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        (self.0 >> 16) & 1 == 1
    }
    #[inline]
    pub fn size(self) -> u8 {
        self.0 as u8
    }
    #[inline]
    pub fn total_consumed(self) -> u8 {
        (self.0 >> 8) as u8
    }
}

/// Combined DC lookup table indexed by the next `FAST_DC_PEEK_WIDTH`
/// bits of the entropy stream.
pub struct FastDcHuffmanTable {
    lut: Box<[FastDcEntry; FAST_DC_LUT_SIZE]>,
}

impl FastDcHuffmanTable {
    /// Build from a DHT spec (DC class). Mirrors `FastAcHuffmanTable::from_spec`
    /// but treats the symbol byte as the magnitude bit count directly.
    pub fn from_spec(spec: &HuffmanTableSpec) -> Result<Self> {
        let (huffsize, huffcode) = build_canonical_huffman(spec)?;

        // Fill the LUT. Symbols whose code_length exceeds PEEK_WIDTH,
        // or whose code_length + size exceeds it, are left at valid=0
        // (the canonical prefix property keeps those slots collision-free).
        let mut lut = Box::new([FastDcEntry::INVALID; FAST_DC_LUT_SIZE]);
        for i in 0..huffsize.len() {
            let l = huffsize[i] as u32;
            if l > FAST_DC_PEEK_WIDTH {
                continue;
            }
            let size = spec.values[i] as u32;
            let total = l + size;
            if total > FAST_DC_PEEK_WIDTH {
                continue;
            }
            let entry = FastDcEntry::new(size as u8, total as u8);
            let base = (huffcode[i] as usize) << (FAST_DC_PEEK_WIDTH - l);
            let span = 1usize << (FAST_DC_PEEK_WIDTH - l);
            for j in 0..span {
                lut[base + j] = entry;
            }
        }

        Ok(Self { lut })
    }

    /// Return the entry for the given PEEK_WIDTH-bit peek key.
    #[inline]
    pub fn lookup(&self, peek_key: u32) -> FastDcEntry {
        self.lut[(peek_key & ((1u32 << FAST_DC_PEEK_WIDTH) - 1)) as usize]
    }

    /// Number of LUT slots flagged valid. Used by the coverage tests.
    #[allow(dead_code)] // diagnostic-only: consumed by the lut-report and coverage tests
    pub fn fast_path_slot_count(&self) -> usize {
        self.lut.iter().filter(|e| e.is_valid()).count()
    }
}

/// One DC term decoded via the combined LUT.
pub struct FastDcDecoded {
    pub size: u8,
    pub magnitude_raw: u32,
    /// Bits consumed on this hit; kept for cross-checking against the slow path.
    #[allow(dead_code)]
    pub total_consumed: u8,
}

/// Combined DC fast-path lookup. Mirrors [`decode_ac_fast`] semantics:
/// returns `Some` on a hit (reader advanced by `total_consumed`),
/// `Ok(None)` on a miss with the bit buffer untouched so the slow
/// path sees the same bits.
pub fn decode_dc_fast(
    br: &mut BitReader,
    tbl: &FastDcHuffmanTable,
) -> Result<Option<FastDcDecoded>> {
    if br.nbits < FAST_DC_PEEK_WIDTH {
        br.fill(FAST_DC_PEEK_WIDTH)?;
    }
    let peek_key =
        ((br.buf >> (br.nbits - FAST_DC_PEEK_WIDTH)) as u32) & ((1u32 << FAST_DC_PEEK_WIDTH) - 1);
    let entry = tbl.lookup(peek_key);
    if !entry.is_valid() {
        return Ok(None);
    }
    let size = entry.size();
    let total = entry.total_consumed();
    let code_length = total - size;
    let magnitude_raw = if size == 0 {
        0
    } else {
        let shift = FAST_DC_PEEK_WIDTH - code_length as u32 - size as u32;
        (peek_key >> shift) & ((1u32 << size) - 1)
    };
    br.drop_bits(total as u32);
    Ok(Some(FastDcDecoded {
        size,
        magnitude_raw,
        total_consumed: total,
    }))
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

    /// Diagnostic-only — print fast-path slot coverage and slow-path
    /// fallback code-length histogram for the standard luma/chroma AC
    /// tables. Marked `#[ignore]` so it doesn't run in CI; invoke as
    /// `cargo test fast_ac_lut_report -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn fast_ac_lut_report() {
        for (name, std, id) in [
            ("luma", &crate::tables::STD_LUMA_AC, 0u8),
            ("chroma", &crate::tables::STD_CHROMA_AC, 1u8),
        ] {
            let spec = HuffmanTableSpec {
                class: 1,
                id,
                bits: std.bits,
                values: std.values.to_vec(),
            };
            let fast = FastAcHuffmanTable::from_spec(&spec).unwrap();
            let valid = fast.fast_path_slot_count();
            let total = 1usize << FAST_AC_PEEK_WIDTH;
            // Reconstruct per-symbol code length so we can classify
            // which symbols stay on the slow path.
            let mut huffsize: Vec<u8> = Vec::with_capacity(spec.values.len());
            for (l_idx, &count) in spec.bits.iter().enumerate() {
                for _ in 0..count {
                    huffsize.push((l_idx + 1) as u8);
                }
            }
            let mut slow_hist = [0usize; 17];
            let mut slow_count = 0usize;
            for (i, &l) in huffsize.iter().enumerate() {
                let sym = spec.values[i];
                let size = (sym & 0x0F) as u32;
                let total_consumed = l as u32 + size;
                if (l as u32) > FAST_AC_PEEK_WIDTH || total_consumed > FAST_AC_PEEK_WIDTH {
                    slow_hist[l as usize] += 1;
                    slow_count += 1;
                }
            }
            eprintln!(
                "[{name} AC] fast-path slots = {valid}/{total} ({:.1}%); slow-path symbols = {slow_count}/{}; slow code-length histogram (l=1..16) = {:?}",
                100.0 * valid as f64 / total as f64,
                spec.values.len(),
                &slow_hist[1..=16],
            );
        }
    }

    fn ac_spec(std: &crate::tables::StdHuffman, id: u8) -> HuffmanTableSpec {
        HuffmanTableSpec {
            class: 1,
            id,
            bits: std.bits,
            values: std.values.to_vec(),
        }
    }

    /// Drive both the combined-LUT fast path and the existing
    /// `decode_symbol` + `get_bits` slow path on the same MSB-aligned
    /// 10-bit input pattern, and require their decoded
    /// (run, size, magnitude_raw) triples agree whenever the LUT
    /// reports a valid entry. Sweeps the full 2^PEEK_WIDTH key space.
    fn cross_check_against_slow_path(spec: &HuffmanTableSpec) -> (usize, usize) {
        let fast = FastAcHuffmanTable::from_spec(spec).unwrap();
        let slow = HuffmanDecodeTable::from_spec(spec).unwrap();
        let mut valid = 0usize;
        for i in 0..(1u32 << FAST_AC_PEEK_WIDTH) {
            // Place `i` as the top PEEK_WIDTH bits of a 32-bit stream.
            let aligned = i << (32 - FAST_AC_PEEK_WIDTH);
            let bytes = aligned.to_be_bytes();
            let mut br_fast = BitReader::new(&bytes, 0);
            let got = decode_ac_fast(&mut br_fast, &fast).unwrap();
            if let Some(d) = got {
                valid += 1;
                assert!(
                    (d.total_consumed as u32) <= FAST_AC_PEEK_WIDTH,
                    "key {i:#x}: total_consumed {} exceeds peek width",
                    d.total_consumed
                );
                // Decode the same stream the slow way and compare.
                let mut br_slow = BitReader::new(&bytes, 0);
                let sym = br_slow.decode_symbol(&slow).unwrap();
                let run_slow = (sym >> 4) & 0x0F;
                let size_slow = sym & 0x0F;
                let mag_slow = if size_slow == 0 {
                    0
                } else {
                    br_slow.get_bits(size_slow as u32).unwrap()
                };
                assert_eq!(d.run, run_slow, "key {i:#x}: run mismatch");
                assert_eq!(d.size, size_slow, "key {i:#x}: size mismatch");
                assert_eq!(
                    d.magnitude_raw, mag_slow,
                    "key {i:#x}: magnitude mismatch (size={size_slow})"
                );
            }
        }
        (valid, 1usize << FAST_AC_PEEK_WIDTH)
    }

    #[test]
    fn fast_ac_lut_round_trips_via_standard_luma_ac() {
        let spec = ac_spec(&crate::tables::STD_LUMA_AC, 0);
        let (valid, total) = cross_check_against_slow_path(&spec);
        // Standard luma AC: codes of length 2..10 contribute ~30 of
        // the 162 symbols and easily cover well over half the
        // 1024-slot key space once magnitude variants are expanded.
        // Pin the lower bound at 50% to catch builder regressions.
        assert!(
            valid * 2 >= total,
            "luma AC fast-path coverage too low: {valid}/{total}"
        );
    }

    #[test]
    fn fast_ac_lut_round_trips_via_standard_chroma_ac() {
        let spec = ac_spec(&crate::tables::STD_CHROMA_AC, 1);
        let (valid, total) = cross_check_against_slow_path(&spec);
        assert!(
            valid * 2 >= total,
            "chroma AC fast-path coverage too low: {valid}/{total}"
        );
    }

    #[test]
    fn fast_ac_lut_marks_long_codes_as_slow_path() {
        // STD_LUMA_AC has 125 length-16 codes — none can fit in a
        // 10-bit peek, so every key that those codes' prefixes occupy
        // must report valid=0. We verify the structural property: no
        // valid entry's recorded code_length exceeds PEEK_WIDTH and no
        // total_consumed exceeds PEEK_WIDTH.
        let spec = ac_spec(&crate::tables::STD_LUMA_AC, 0);
        let fast = FastAcHuffmanTable::from_spec(&spec).unwrap();
        for i in 0..(1u32 << FAST_AC_PEEK_WIDTH) {
            let e = fast.lookup(i);
            if e.is_valid() {
                let total = e.total_consumed() as u32;
                let size = e.size() as u32;
                assert!(total <= FAST_AC_PEEK_WIDTH);
                assert!(size <= total);
            }
        }
    }

    #[test]
    fn fast_ac_lut_handles_zrl_and_eob() {
        // ZRL = 0xF0 (run=15, size=0); EOB = 0x00 (run=0, size=0).
        // Both are short codes in the standard luma AC table, so they
        // must land in the fast path with size==0 and total_consumed
        // equal to their code length (no magnitude bits).
        let spec = ac_spec(&crate::tables::STD_LUMA_AC, 0);
        let fast = FastAcHuffmanTable::from_spec(&spec).unwrap();

        // EOB code is 1010 (length 4) per Annex K.4; ZRL is
        // 11111111001 (length 11) — outside our 10-bit window, so
        // ZRL stays slow-path. EOB must be fast-path.
        let mut found_eob = false;
        let mut found_run0_size_nonzero = false;
        for i in 0..(1u32 << FAST_AC_PEEK_WIDTH) {
            let e = fast.lookup(i);
            if e.is_valid() {
                if e.run() == 0 && e.size() == 0 {
                    found_eob = true;
                }
                if e.run() == 0 && e.size() != 0 {
                    found_run0_size_nonzero = true;
                }
            }
        }
        assert!(found_eob, "EOB (run=0, size=0) must hit fast path");
        assert!(
            found_run0_size_nonzero,
            "expected at least one run=0 non-zero size symbol in fast path"
        );
    }

    fn dc_spec(std: &crate::tables::StdHuffman, id: u8) -> HuffmanTableSpec {
        HuffmanTableSpec {
            class: 0,
            id,
            bits: std.bits,
            values: std.values.to_vec(),
        }
    }

    /// Diagnostic-only — print fast-path slot coverage for the standard
    /// luma/chroma DC tables. Marked `#[ignore]` so it doesn't run in CI;
    /// invoke as
    /// `cargo test fast_dc_lut_report -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn fast_dc_lut_report() {
        for (name, std, id) in [
            ("luma", &crate::tables::STD_LUMA_DC, 0u8),
            ("chroma", &crate::tables::STD_CHROMA_DC, 1u8),
        ] {
            let spec = HuffmanTableSpec {
                class: 0,
                id,
                bits: std.bits,
                values: std.values.to_vec(),
            };
            let fast = FastDcHuffmanTable::from_spec(&spec).unwrap();
            let valid = fast.fast_path_slot_count();
            let total = 1usize << FAST_DC_PEEK_WIDTH;
            let mut huffsize: Vec<u8> = Vec::with_capacity(spec.values.len());
            for (l_idx, &count) in spec.bits.iter().enumerate() {
                for _ in 0..count {
                    huffsize.push((l_idx + 1) as u8);
                }
            }
            let mut slow_count = 0usize;
            for (i, &l) in huffsize.iter().enumerate() {
                let size = spec.values[i] as u32;
                let total_consumed = l as u32 + size;
                if (l as u32) > FAST_DC_PEEK_WIDTH || total_consumed > FAST_DC_PEEK_WIDTH {
                    slow_count += 1;
                }
            }
            eprintln!(
                "[{name} DC] fast-path slots = {valid}/{total} ({:.1}%); slow-path symbols = {slow_count}/{}",
                100.0 * valid as f64 / total as f64,
                spec.values.len(),
            );
        }
    }

    /// Drive the combined DC fast path and the canonical two-step path
    /// (decode_symbol then get_bits) on the same MSB-aligned 10-bit
    /// input, and require their (size, magnitude_raw) pairs agree on
    /// every LUT hit.
    fn cross_check_dc_against_slow_path(spec: &HuffmanTableSpec) -> (usize, usize) {
        let fast = FastDcHuffmanTable::from_spec(spec).unwrap();
        let slow = HuffmanDecodeTable::from_spec(spec).unwrap();
        let mut valid = 0usize;
        for i in 0..(1u32 << FAST_DC_PEEK_WIDTH) {
            let aligned = i << (32 - FAST_DC_PEEK_WIDTH);
            let bytes = aligned.to_be_bytes();
            let mut br_fast = BitReader::new(&bytes, 0);
            let got = decode_dc_fast(&mut br_fast, &fast).unwrap();
            if let Some(d) = got {
                valid += 1;
                assert!(
                    (d.total_consumed as u32) <= FAST_DC_PEEK_WIDTH,
                    "key {i:#x}: total_consumed {} exceeds peek width",
                    d.total_consumed
                );
                let mut br_slow = BitReader::new(&bytes, 0);
                let sym = br_slow.decode_symbol(&slow).unwrap();
                let size_slow = sym;
                let mag_slow = if size_slow == 0 {
                    0
                } else {
                    br_slow.get_bits(size_slow as u32).unwrap()
                };
                assert_eq!(d.size, size_slow, "key {i:#x}: size mismatch");
                assert_eq!(
                    d.magnitude_raw, mag_slow,
                    "key {i:#x}: magnitude mismatch (size={size_slow})"
                );
            }
        }
        (valid, 1usize << FAST_DC_PEEK_WIDTH)
    }

    #[test]
    fn fast_dc_lut_round_trips_via_standard_luma_dc() {
        let spec = dc_spec(&crate::tables::STD_LUMA_DC, 0);
        let (valid, total) = cross_check_dc_against_slow_path(&spec);
        // DC codes are short (length 2-9 for standard luma) and sizes
        // 0..=7 dominate, so the fast path should cover well over half
        // the 1024-slot key space.
        assert!(
            valid * 2 >= total,
            "luma DC fast-path coverage too low: {valid}/{total}"
        );
    }

    #[test]
    fn fast_dc_lut_round_trips_via_standard_chroma_dc() {
        let spec = dc_spec(&crate::tables::STD_CHROMA_DC, 1);
        let (valid, total) = cross_check_dc_against_slow_path(&spec);
        assert!(
            valid * 2 >= total,
            "chroma DC fast-path coverage too low: {valid}/{total}"
        );
    }

    #[test]
    fn fast_dc_lut_marks_oversize_as_slow_path() {
        // Structural invariant: every valid entry's total_consumed and
        // size fits within PEEK_WIDTH; anything larger (size 8..=11
        // combined with non-trivial code length) must land slow-path.
        let spec = dc_spec(&crate::tables::STD_LUMA_DC, 0);
        let fast = FastDcHuffmanTable::from_spec(&spec).unwrap();
        for i in 0..(1u32 << FAST_DC_PEEK_WIDTH) {
            let e = fast.lookup(i);
            if e.is_valid() {
                let total = e.total_consumed() as u32;
                let size = e.size() as u32;
                assert!(total <= FAST_DC_PEEK_WIDTH);
                assert!(size <= total);
            }
        }
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
