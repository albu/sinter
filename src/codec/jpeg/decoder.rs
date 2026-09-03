use crate::codec::jpeg::color::ycbcr_to_rgb;
use crate::codec::jpeg::error::JpegError;
use crate::codec::jpeg::huffman::{BitReader, HuffmanTable};
use crate::codec::jpeg::idct::idct_8x8;
use crate::codec::jpeg::marker::{ComponentInfo, JpegHeader, ZIG_ZAG};

pub struct JpegDecoder<'a> {
    pub data: &'a [u8],
    pub header: Option<JpegHeader>,
    pub dc_tables: [Option<HuffmanTable>; 4],
    pub ac_tables: [Option<HuffmanTable>; 4],
}

impl<'a> JpegDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            header: None,
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
        }
    }

    /// Parse JPEG markers up to Start of Scan (SOS)
    pub fn parse_header(&mut self) -> Result<&JpegHeader, JpegError> {
        if self.header.is_some() {
            return Ok(self.header.as_ref().unwrap());
        }

        if self.data.len() < 4 || self.data[0] != 0xFF || self.data[1] != 0xD8 {
            return Err(JpegError::InvalidMarker(
                if self.data.len() >= 2 { ((self.data[0] as u16) << 8) | (self.data[1] as u16) } else { 0 }
            ));
        }

        let mut pos = 2;
        let mut width = 0;
        let mut height = 0;
        let mut precision = 8;
        let mut components = Vec::new();
        let mut max_h_samp = 1;
        let mut max_v_samp = 1;
        let mut dqt = [[0u16; 64]; 4];
        let mut has_dqt = [false; 4];
        let mut restart_interval = 0;
        let mut scan_offset = 0;
        let mut seen_sof = false;

        while pos + 1 < self.data.len() {
            if self.data[pos] != 0xFF {
                pos += 1;
                continue;
            }

            let marker = self.data[pos + 1];
            pos += 2;

            // Skip padding 0xFF bytes
            if marker == 0xFF || marker == 0x00 {
                continue;
            }

            match marker {
                0xD8 => continue, // SOI
                0xD9 => break,    // EOI
                0xDB => {
                    // DQT: Define Quantization Table
                    if pos + 2 > self.data.len() { return Err(JpegError::UnexpectedEof); }
                    let len = (((self.data[pos] as usize) << 8) | (self.data[pos + 1] as usize)) - 2;
                    pos += 2;

                    let mut bytes_read = 0;
                    while bytes_read < len && pos < self.data.len() {
                        let info = self.data[pos];
                        pos += 1;
                        bytes_read += 1;

                        let precision_dqt = info >> 4;
                        let table_id = (info & 0x0F) as usize;
                        if table_id >= 4 {
                            return Err(JpegError::CorruptedBitstream("Invalid DQT table ID".into()));
                        }

                        if precision_dqt == 0 {
                            // 8-bit precision
                            for i in 0..64 {
                                if pos >= self.data.len() { return Err(JpegError::UnexpectedEof); }
                                dqt[table_id][i] = self.data[pos] as u16;
                                pos += 1;
                            }
                            bytes_read += 64;
                        } else {
                            // 16-bit precision
                            for i in 0..64 {
                                if pos + 1 >= self.data.len() { return Err(JpegError::UnexpectedEof); }
                                dqt[table_id][i] = ((self.data[pos] as u16) << 8) | (self.data[pos + 1] as u16);
                                pos += 2;
                            }
                            bytes_read += 128;
                        }
                        has_dqt[table_id] = true;
                    }
                }
                0xC0 | 0xC2 => {
                    // SOF0: Baseline DCT or SOF2: Progressive DCT
                    if pos + 2 > self.data.len() { return Err(JpegError::UnexpectedEof); }
                    let _len = ((self.data[pos] as usize) << 8) | (self.data[pos + 1] as usize);
                    pos += 2;

                    precision = self.data[pos];
                    if precision != 8 {
                        return Err(JpegError::UnsupportedPrecision(precision));
                    }
                    height = (((self.data[pos + 1] as usize) << 8) | (self.data[pos + 2] as usize));
                    width = (((self.data[pos + 3] as usize) << 8) | (self.data[pos + 4] as usize));
                    let num_components = self.data[pos + 5] as usize;
                    pos += 6;

                    if num_components != 1 && num_components != 3 {
                        return Err(JpegError::UnsupportedComponentCount(num_components));
                    }

                    components.clear();
                    for _ in 0..num_components {
                        let id = self.data[pos];
                        let sampling = self.data[pos + 1];
                        let h_samp = sampling >> 4;
                        let v_samp = sampling & 0x0F;
                        let quant_tbl_id = self.data[pos + 2];
                        pos += 3;

                        if h_samp > max_h_samp { max_h_samp = h_samp; }
                        if v_samp > max_v_samp { max_v_samp = v_samp; }

                        components.push(ComponentInfo {
                            id,
                            h_samp,
                            v_samp,
                            quant_tbl_id,
                            dc_tbl_id: 0,
                            ac_tbl_id: 0,
                        });
                    }
                    seen_sof = true;
                }
                0xC4 => {
                    // DHT: Define Huffman Table
                    if pos + 2 > self.data.len() { return Err(JpegError::UnexpectedEof); }
                    let len = (((self.data[pos] as usize) << 8) | (self.data[pos + 1] as usize)) - 2;
                    pos += 2;

                    let mut bytes_read = 0;
                    while bytes_read < len && pos < self.data.len() {
                        let info = self.data[pos];
                        pos += 1;
                        bytes_read += 1;

                        let class = (info >> 4) as usize; // 0 = DC, 1 = AC
                        let table_id = (info & 0x0F) as usize;
                        if table_id >= 4 {
                            return Err(JpegError::CorruptedBitstream("Invalid DHT table ID".into()));
                        }

                        let mut counts = [0u8; 16];
                        for i in 0..16 {
                            counts[i] = self.data[pos];
                            pos += 1;
                        }
                        bytes_read += 16;

                        let sym_count: usize = counts.iter().map(|&c| c as usize).sum();
                        let symbols = &self.data[pos..pos + sym_count];
                        pos += sym_count;
                        bytes_read += sym_count;

                        let table = HuffmanTable::build(&counts, symbols)?;
                        if class == 0 {
                            self.dc_tables[table_id] = Some(table);
                        } else {
                            self.ac_tables[table_id] = Some(table);
                        }
                    }
                }
                0xDD => {
                    // DRI: Define Restart Interval
                    if pos + 4 > self.data.len() { return Err(JpegError::UnexpectedEof); }
                    restart_interval = ((self.data[pos + 2] as u16) << 8) | (self.data[pos + 3] as u16);
                    pos += 4;
                }
                0xDA => {
                    // SOS: Start of Scan
                    if pos + 2 > self.data.len() { return Err(JpegError::UnexpectedEof); }
                    let len = ((self.data[pos] as usize) << 8) | (self.data[pos + 1] as usize);
                    pos += 2;

                    let num_scan_comp = self.data[pos] as usize;
                    pos += 1;

                    for _ in 0..num_scan_comp {
                        let comp_id = self.data[pos];
                        let tbl_sel = self.data[pos + 1];
                        let dc_id = tbl_sel >> 4;
                        let ac_id = tbl_sel & 0x0F;
                        pos += 2;

                        for comp in &mut components {
                            if comp.id == comp_id {
                                comp.dc_tbl_id = dc_id;
                                comp.ac_tbl_id = ac_id;
                            }
                        }
                    }

                    // Skip spectral selection and approx bytes (3 bytes)
                    pos += 3;
                    scan_offset = pos;
                    break;
                }
                _ => {
                    // Skip other markers (APPn, COM, etc.) by reading segment length
                    if pos + 2 <= self.data.len() {
                        let len = ((self.data[pos] as usize) << 8) | (self.data[pos + 1] as usize);
                        pos += len;
                    }
                }
            }
        }

        if !seen_sof {
            return Err(JpegError::MissingSOF);
        }
        if scan_offset == 0 {
            return Err(JpegError::MissingSOS);
        }

        self.header = Some(JpegHeader {
            width,
            height,
            precision,
            components,
            max_h_samp,
            max_v_samp,
            dqt,
            has_dqt,
            restart_interval,
            scan_offset,
        });

        Ok(self.header.as_ref().unwrap())
    }

    /// Decode the full image directly to RGB8
    pub fn decode(&mut self) -> Result<(usize, usize, usize, Vec<u8>), JpegError> {
        let (w, h) = {
            let hdr = self.parse_header()?;
            (hdr.width, hdr.height)
        };
        self.decode_crop(0, 0, w, h)
    }

    /// Decode only a specified Region of Interest (ROI) crop.
    /// Skips MCU dequantization, IDCT, and color conversion outside the crop bounding box!
    pub fn decode_crop(
        &mut self,
        crop_x: usize,
        crop_y: usize,
        crop_w: usize,
        crop_h: usize,
    ) -> Result<(usize, usize, usize, Vec<u8>), JpegError> {
        self.parse_header()?;
        let header = self.header.as_ref().unwrap().clone();

        if crop_x + crop_w > header.width || crop_y + crop_h > header.height {
            return Err(JpegError::CropOutOfBounds);
        }

        let mcu_w = header.mcu_width();
        let mcu_h = header.mcu_height();
        let mcus_x = header.mcus_x();
        let mcus_y = header.mcus_y();

        // Determine bounding box of MCUs overlapping the crop region
        let mcu_start_x = crop_x / mcu_w;
        let mcu_end_x = (crop_x + crop_w + mcu_w - 1) / mcu_w;
        let mcu_start_y = crop_y / mcu_h;
        let mcu_end_y = (crop_y + crop_h + mcu_h - 1) / mcu_h;

        let num_components = header.components.len();
        let mut prev_dc = vec![0i16; num_components];
        let mut reader = BitReader::new(&self.data[header.scan_offset..]);

        let mut out_rgb = vec![0u8; crop_w * crop_h * 3];

        // Workspace buffers
        let mut coeff_block = [0i32; 64];
        let mut y_pixels = vec![0u8; mcu_w * mcu_h];
        let mut cb_pixels = vec![0u8; mcu_w * mcu_h];
        let mut cr_pixels = vec![0u8; mcu_w * mcu_h];

        let mut mcu_count = 0;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                // Check restart interval
                if header.restart_interval > 0 && mcu_count > 0 && (mcu_count % (header.restart_interval as usize)) == 0 {
                    reader.refill();
                    reader.reset_buffer();
                    prev_dc.fill(0);
                }
                mcu_count += 1;

                let inside_roi = mcu_x >= mcu_start_x && mcu_x < mcu_end_x && mcu_y >= mcu_start_y && mcu_y < mcu_end_y;

                if !inside_roi {
                    // FAST PATH: Skip MCU dequantization, IDCT, and color conversion!
                    for (c_idx, comp) in header.components.iter().enumerate() {
                        let dc_table = self.dc_tables[comp.dc_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;
                        let ac_table = self.ac_tables[comp.ac_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;

                        let blocks_in_comp = (comp.h_samp as usize) * (comp.v_samp as usize);
                        for _ in 0..blocks_in_comp {
                            reader.skip_block(dc_table, ac_table, &mut prev_dc[c_idx])?;
                        }
                    }
                    continue;
                }

                // SLOW PATH: Decode MCU, run IDCT, and write intersection into crop buffer
                let mcu_px_x = mcu_x * mcu_w;
                let mcu_px_y = mcu_y * mcu_h;

                if num_components == 3 {
                    // Y component blocks
                    let comp_y = &header.components[0];
                    let dc_y = self.dc_tables[comp_y.dc_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;
                    let ac_y = self.ac_tables[comp_y.ac_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;
                    let q_y = &header.dqt[comp_y.quant_tbl_id as usize];

                    for v in 0..(comp_y.v_samp as usize) {
                        for h in 0..(comp_y.h_samp as usize) {
                            let has_ac = reader.decode_block(dc_y, ac_y, q_y, &mut prev_dc[0], &mut coeff_block)?;
                            let mut block_u8 = [0u8; 64];
                            if has_ac {
                                idct_8x8(&coeff_block, &mut block_u8);
                            } else {
                                crate::codec::jpeg::idct::idct_8x8_dc(coeff_block[0], &mut block_u8);
                            }

                            for row in 0..8 {
                                let dst_y = v * 8 + row;
                                let dst_x = h * 8;
                                let src_offset = row * 8;
                                let dst_offset = dst_y * mcu_w + dst_x;
                                y_pixels[dst_offset..dst_offset + 8].copy_from_slice(&block_u8[src_offset..src_offset + 8]);
                            }
                        }
                    }

                    // Cb component
                    let comp_cb = &header.components[1];
                    let dc_cb = self.dc_tables[comp_cb.dc_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;
                    let ac_cb = self.ac_tables[comp_cb.ac_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;
                    let q_cb = &header.dqt[comp_cb.quant_tbl_id as usize];
                    let has_ac_cb = reader.decode_block(dc_cb, ac_cb, q_cb, &mut prev_dc[1], &mut coeff_block)?;
                    let mut cb_block_u8 = [0u8; 64];
                    if has_ac_cb {
                        idct_8x8(&coeff_block, &mut cb_block_u8);
                    } else {
                        crate::codec::jpeg::idct::idct_8x8_dc(coeff_block[0], &mut cb_block_u8);
                    }

                    // Cr component
                    let comp_cr = &header.components[2];
                    let dc_cr = self.dc_tables[comp_cr.dc_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;
                    let ac_cr = self.ac_tables[comp_cr.ac_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;
                    let q_cr = &header.dqt[comp_cr.quant_tbl_id as usize];
                    let has_ac_cr = reader.decode_block(dc_cr, ac_cr, q_cr, &mut prev_dc[2], &mut coeff_block)?;
                    let mut cr_block_u8 = [0u8; 64];
                    if has_ac_cr {
                        idct_8x8(&coeff_block, &mut cr_block_u8);
                    } else {
                        crate::codec::jpeg::idct::idct_8x8_dc(coeff_block[0], &mut cr_block_u8);
                    }

                    // Fast chroma upsampling (4:2:0 standard)
                    if comp_y.h_samp == 2 && comp_y.v_samp == 2 {
                        for cy in 0..8 {
                            let src_cb = &cb_block_u8[cy * 8..(cy + 1) * 8];
                            let src_cr = &cr_block_u8[cy * 8..(cy + 1) * 8];
                            let mut dup_cb = [0u8; 16];
                            let mut dup_cr = [0u8; 16];
                            for i in 0..8 {
                                dup_cb[i * 2] = src_cb[i];
                                dup_cb[i * 2 + 1] = src_cb[i];
                                dup_cr[i * 2] = src_cr[i];
                                dup_cr[i * 2 + 1] = src_cr[i];
                            }
                            let dst_row0 = (cy * 2) * 16;
                            let dst_row1 = (cy * 2 + 1) * 16;
                            cb_pixels[dst_row0..dst_row0 + 16].copy_from_slice(&dup_cb);
                            cb_pixels[dst_row1..dst_row1 + 16].copy_from_slice(&dup_cb);
                            cr_pixels[dst_row0..dst_row0 + 16].copy_from_slice(&dup_cr);
                            cr_pixels[dst_row1..dst_row1 + 16].copy_from_slice(&dup_cr);
                        }
                    } else {
                        // 4:4:4 or 1:1 sampling
                        cb_pixels[..64].copy_from_slice(&cb_block_u8);
                        cr_pixels[..64].copy_from_slice(&cr_block_u8);
                    }

                    // Write rows into crop_buffer (vectorized where possible)
                    for py in 0..mcu_h {
                        let img_y = mcu_px_y + py;
                        if img_y < crop_y || img_y >= crop_y + crop_h {
                            continue;
                        }
                        let out_y = img_y - crop_y;

                        // Fast path: MCU row is completely within horizontal crop bounds
                        if mcu_px_x >= crop_x && mcu_px_x + mcu_w <= crop_x + crop_w {
                            let out_x = mcu_px_x - crop_x;
                            let mcu_offset = py * mcu_w;
                            let out_start = (out_y * crop_w + out_x) * 3;

                            #[cfg(target_arch = "aarch64")]
                            if mcu_w == 16 {
                                unsafe {
                                    crate::codec::jpeg::color::ycbcr_to_rgb_16(
                                        y_pixels.as_ptr().add(mcu_offset),
                                        cb_pixels.as_ptr().add(mcu_offset),
                                        cr_pixels.as_ptr().add(mcu_offset),
                                        out_rgb.as_mut_ptr().add(out_start),
                                    );
                                }
                            } else {
                                let y_row = &y_pixels[mcu_offset..mcu_offset + mcu_w];
                                let cb_row = &cb_pixels[mcu_offset..mcu_offset + mcu_w];
                                let cr_row = &cr_pixels[mcu_offset..mcu_offset + mcu_w];
                                let out_slice = &mut out_rgb[out_start..out_start + mcu_w * 3];
                                crate::codec::jpeg::color::ycbcr_to_rgb_slice(y_row, cb_row, cr_row, out_slice);
                            }

                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                let y_row = &y_pixels[mcu_offset..mcu_offset + mcu_w];
                                let cb_row = &cb_pixels[mcu_offset..mcu_offset + mcu_w];
                                let cr_row = &cr_pixels[mcu_offset..mcu_offset + mcu_w];
                                let out_slice = &mut out_rgb[out_start..out_start + mcu_w * 3];
                                crate::codec::jpeg::color::ycbcr_to_rgb_slice(y_row, cb_row, cr_row, out_slice);
                            }
                        } else {
                            // Boundary pixels: scalar fallback
                            for px in 0..mcu_w {
                                let img_x = mcu_px_x + px;
                                if img_x < crop_x || img_x >= crop_x + crop_w {
                                    continue;
                                }
                                let out_x = img_x - crop_x;

                                let mcu_idx = py * mcu_w + px;
                                let (r, g, b) = ycbcr_to_rgb(y_pixels[mcu_idx], cb_pixels[mcu_idx], cr_pixels[mcu_idx]);

                                let out_idx = (out_y * crop_w + out_x) * 3;
                                out_rgb[out_idx] = r;
                                out_rgb[out_idx + 1] = g;
                                out_rgb[out_idx + 2] = b;
                            }
                        }
                    }
                } else {
                    // Grayscale 1-component
                    let comp_y = &header.components[0];
                    let dc_y = self.dc_tables[comp_y.dc_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;
                    let ac_y = self.ac_tables[comp_y.ac_tbl_id as usize].as_ref().ok_or(JpegError::MissingDHT)?;
                    let q_y = &header.dqt[comp_y.quant_tbl_id as usize];

                    reader.decode_block(dc_y, ac_y, q_y, &mut prev_dc[0], &mut coeff_block)?;
                    let mut block_u8 = [0u8; 64];
                    idct_8x8(&coeff_block, &mut block_u8);

                    for py in 0..8 {
                        let img_y = mcu_px_y + py;
                        if img_y < crop_y || img_y >= crop_y + crop_h {
                            continue;
                        }
                        let out_y = img_y - crop_y;

                        for px in 0..8 {
                            let img_x = mcu_px_x + px;
                            if img_x < crop_x || img_x >= crop_x + crop_w {
                                continue;
                            }
                            let out_x = img_x - crop_x;
                            let val = block_u8[py * 8 + px];

                            let out_idx = (out_y * crop_w + out_x) * 3;
                            out_rgb[out_idx] = val;
                            out_rgb[out_idx + 1] = val;
                            out_rgb[out_idx + 2] = val;
                        }
                    }
                }
            }
        }

        Ok((crop_w, crop_h, 3, out_rgb))
    }
}
