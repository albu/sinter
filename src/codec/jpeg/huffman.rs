use crate::codec::jpeg::error::JpegError;
use crate::codec::jpeg::marker::ZIG_ZAG;

/// Fast 10-bit lookup table (1024 entries) + canonical JPEG Huffman decoder
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    pub lut: [i16; 1024],
    pub maxcode: [i32; 18],
    pub mincode: [i32; 18],
    pub valptr: [usize; 18],
    pub symbols: Vec<u8>,
}

impl Default for HuffmanTable {
    fn default() -> Self {
        Self {
            lut: [-1; 1024],
            maxcode: [-1; 18],
            mincode: [-1; 18],
            valptr: [0; 18],
            symbols: Vec::new(),
        }
    }
}

impl HuffmanTable {
    /// Build fast lookup table from JPEG DHT bits counts (16 bytes) and symbol values
    pub fn build(counts: &[u8; 16], symbols: &[u8]) -> Result<Self, JpegError> {
        let mut table = Self {
            lut: [-1; 1024],
            maxcode: [-1; 18],
            mincode: [-1; 18],
            valptr: [0; 18],
            symbols: symbols.to_vec(),
        };

        let mut code: i32 = 0;
        let mut sym_idx = 0;

        for l in 1..=16 {
            let count = counts[l - 1] as i32;
            if count == 0 {
                table.maxcode[l] = -1;
                table.mincode[l] = -1;
            } else {
                table.valptr[l] = sym_idx;
                table.mincode[l] = code;

                for _ in 0..count {
                    if sym_idx >= symbols.len() {
                        return Err(JpegError::CorruptedBitstream("Huffman symbols overflow".into()));
                    }
                    let symbol = symbols[sym_idx];

                    if l <= 10 {
                        let shift = 10 - l;
                        let start = ((code as usize) << shift);
                        let end = start + (1 << shift);
                        let entry = ((l as i16) << 8) | (symbol as i16);
                        for i in start..end {
                            table.lut[i] = entry;
                        }
                    }
                    sym_idx += 1;
                    code += 1;
                }
                table.maxcode[l] = code - 1;
            }
            code <<= 1;
        }

        Ok(table)
    }
}

/// Bitstream reader supporting byte stuffing (0xFF 0x00 -> 0xFF) and restart markers
pub struct BitReader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
    pub bit_buf: u64,
    pub bits_left: u32,
    pub marker: Option<u8>,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bits_left: 0,
            marker: None,
        }
    }

    /// Reset bit buffer state (e.g. after restart marker)
    pub fn reset_buffer(&mut self) {
        self.bit_buf = 0;
        self.bits_left = 0;
        self.marker = None;
    }

    /// Refill the 64-bit bit buffer to have at least 32 bits available
    #[inline(always)]
    pub fn refill(&mut self) {
        // Fast path: bulk load 4 bytes (32 bits) at once if no 0xFF byte present
        while self.bits_left <= 32 && self.pos + 4 <= self.data.len() {
            let b0 = self.data[self.pos];
            let b1 = self.data[self.pos + 1];
            let b2 = self.data[self.pos + 2];
            let b3 = self.data[self.pos + 3];

            if b0 != 0xFF && b1 != 0xFF && b2 != 0xFF && b3 != 0xFF {
                let word = ((b0 as u64) << 24) | ((b1 as u64) << 16) | ((b2 as u64) << 8) | (b3 as u64);
                self.bit_buf = (self.bit_buf << 32) | word;
                self.bits_left += 32;
                self.pos += 4;
            } else {
                break;
            }
        }

        while self.bits_left <= 48 && self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;

            if b == 0xFF {
                if self.pos < self.data.len() {
                    let b2 = self.data[self.pos];
                    self.pos += 1;
                    if b2 == 0x00 {
                        self.bit_buf = (self.bit_buf << 8) | 0xFF;
                        self.bits_left += 8;
                    } else if (0xD0..=0xD7).contains(&b2) {
                        self.marker = Some(b2);
                        break;
                    } else {
                        self.marker = Some(b2);
                        break;
                    }
                } else {
                    self.bit_buf = (self.bit_buf << 8) | 0xFF;
                    self.bits_left += 8;
                }
            } else {
                self.bit_buf = (self.bit_buf << 8) | (b as u64);
                self.bits_left += 8;
            }
        }
    }

    #[inline(always)]
    pub fn ensure_bits(&mut self, n: u32) {
        if self.bits_left < n {
            self.refill();
        }
    }

    #[inline(always)]
    pub fn peek_bits_unchecked(&self, n: u32) -> u16 {
        ((self.bit_buf >> (self.bits_left - n)) & ((1u64 << n) - 1)) as u16
    }

    #[inline(always)]
    pub fn consume_bits_unchecked(&mut self, n: u32) {
        self.bits_left -= n;
        self.bit_buf &= (1u64 << self.bits_left) - 1;
    }

    #[inline(always)]
    pub fn read_bits_unchecked(&mut self, n: u32) -> u16 {
        if n == 0 {
            return 0;
        }
        self.bits_left -= n;
        let val = ((self.bit_buf >> self.bits_left) & ((1u64 << n) - 1)) as u16;
        self.bit_buf &= (1u64 << self.bits_left) - 1;
        val
    }

    #[inline(always)]
    pub fn peek_bits(&mut self, n: u8) -> u16 {
        if self.bits_left < (n as u32) {
            self.refill();
        }
        if self.bits_left == 0 {
            return 0;
        }
        let n = (n as u32).min(self.bits_left);
        ((self.bit_buf >> (self.bits_left - n)) & ((1u64 << n) - 1)) as u16
    }

    #[inline(always)]
    pub fn consume_bits(&mut self, n: u8) {
        if self.bits_left < (n as u32) {
            self.refill();
        }
        let n = (n as u32).min(self.bits_left);
        self.bits_left -= n;
        self.bit_buf &= (1u64 << self.bits_left) - 1;
    }

    #[inline(always)]
    pub fn read_bits(&mut self, n: u8) -> u16 {
        if n == 0 {
            return 0;
        }
        if self.bits_left < (n as u32) {
            self.refill();
        }
        if self.bits_left == 0 {
            return 0;
        }
        let n = (n as u32).min(self.bits_left);
        let val = ((self.bit_buf >> (self.bits_left - n)) & ((1u64 << n) - 1)) as u16;
        self.bits_left -= n;
        self.bit_buf &= (1u64 << self.bits_left) - 1;
        val
    }

    /// Decode a single symbol from a Huffman table
    #[inline(always)]
    pub fn decode_huffman(&mut self, table: &HuffmanTable) -> Result<u8, JpegError> {
        self.ensure_bits(16);
        let peek = self.peek_bits_unchecked(10.min(self.bits_left)) as usize;
        let entry = table.lut[peek];

        if entry >= 0 {
            let length = (entry >> 8) as u32;
            let symbol = (entry & 0xFF) as u8;
            self.consume_bits_unchecked(length);
            Ok(symbol)
        } else {
            self.decode_huffman_slow(table)
        }
    }

    #[inline(never)]
    pub fn decode_huffman_slow(&mut self, table: &HuffmanTable) -> Result<u8, JpegError> {
        for l in 11..=16 {
            let code = self.peek_bits(l as u8) as i32;
            if code <= table.maxcode[l] {
                self.consume_bits(l as u8);
                let idx = table.valptr[l] + (code - table.mincode[l]) as usize;
                if idx < table.symbols.len() {
                    return Ok(table.symbols[idx]);
                }
                break;
            }
        }
        Err(JpegError::CorruptedBitstream("Invalid Huffman code in bitstream".into()))
    }

    /// Decode DC coefficient difference
    #[inline(always)]
    pub fn decode_dc(&mut self, dc_table: &HuffmanTable, prev_dc: &mut i16) -> Result<(), JpegError> {
        let s = self.decode_huffman(dc_table)?;
        if s > 0 {
            self.ensure_bits(s as u32);
            let raw = self.read_bits_unchecked(s as u32) as i32;
            let diff = if raw < (1 << (s - 1)) {
                raw - ((1 << s) - 1)
            } else {
                raw
            };
            *prev_dc += diff as i16;
        }
        Ok(())
    }

    /// Decode an entire 8x8 block (dequantized into output array)
    #[inline(always)]
    pub fn decode_block(
        &mut self,
        dc_table: &HuffmanTable,
        ac_table: &HuffmanTable,
        quant_table: &[u16; 64],
        prev_dc: &mut i16,
        block: &mut [i32; 64],
    ) -> Result<bool, JpegError> {
        block.fill(0);

        self.decode_dc(dc_table, prev_dc)?;
        block[0] = (*prev_dc as i32) * (quant_table[0] as i32);

        let mut k = 1;
        let mut has_ac = false;
        while k < 64 {
            self.ensure_bits(32);
            let peek = self.peek_bits_unchecked(10.min(self.bits_left)) as usize;
            let entry = ac_table.lut[peek];

            if entry >= 0 {
                let length = (entry >> 8) as u32;
                let s = (entry & 0xFF) as u8;
                let size = (s & 0x0F) as u32;
                let run = s >> 4;

                if size == 0 {
                    if run == 0 {
                        self.consume_bits_unchecked(length);
                        break; // EOB
                    } else if run == 15 {
                        self.consume_bits_unchecked(length);
                        k += 16;
                        continue;
                    }
                }

                self.consume_bits_unchecked(length);
                k += run as usize;
                if k >= 64 {
                    break;
                }

                let raw = self.read_bits_unchecked(size) as i32;
                let val = if raw < (1 << (size - 1)) {
                    raw - ((1 << size) - 1)
                } else {
                    raw
                };

                let natural_idx = ZIG_ZAG[k];
                block[natural_idx] = val * (quant_table[natural_idx] as i32);
                has_ac = true;
                k += 1;
            } else {
                let s = self.decode_huffman_slow(ac_table)?;
                let run = s >> 4;
                let size = s & 0x0F;

                if size == 0 {
                    if run == 0 {
                        break;
                    } else if run == 15 {
                        k += 16;
                        continue;
                    }
                }

                k += run as usize;
                if k >= 64 {
                    break;
                }

                let raw = self.read_bits(size) as i32;
                let val = if raw < (1 << (size - 1)) {
                    raw - ((1 << size) - 1)
                } else {
                    raw
                };

                let natural_idx = ZIG_ZAG[k];
                block[natural_idx] = val * (quant_table[natural_idx] as i32);
                has_ac = true;
                k += 1;
            }
        }

        Ok(has_ac)
    }

    /// Skip an 8x8 block rapidly: maintains DC prediction but skips AC storage & IDCT
    #[inline(always)]
    pub fn skip_block(
        &mut self,
        dc_table: &HuffmanTable,
        ac_table: &HuffmanTable,
        prev_dc: &mut i16,
    ) -> Result<(), JpegError> {
        self.decode_dc(dc_table, prev_dc)?;

        let mut k = 1;
        while k < 64 {
            self.ensure_bits(32);
            let peek = self.peek_bits_unchecked(10.min(self.bits_left)) as usize;
            let entry = ac_table.lut[peek];

            if entry >= 0 {
                let length = (entry >> 8) as u32;
                let s = (entry & 0xFF) as u8;
                let size = (s & 0x0F) as u32;
                let run = s >> 4;

                if size == 0 {
                    if run == 0 {
                        self.consume_bits_unchecked(length);
                        break;
                    } else if run == 15 {
                        self.consume_bits_unchecked(length);
                        k += 16;
                        continue;
                    }
                }

                self.consume_bits_unchecked(length + size);
                k += (run as usize) + 1;
            } else {
                let s = self.decode_huffman_slow(ac_table)?;
                let run = s >> 4;
                let size = s & 0x0F;

                if size == 0 {
                    if run == 0 {
                        break;
                    } else if run == 15 {
                        k += 16;
                        continue;
                    }
                }

                k += (run as usize) + 1;
                if size > 0 {
                    self.consume_bits(size);
                }
            }
        }

        Ok(())
    }
}
