pub const ZIG_ZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentInfo {
    pub id: u8,
    pub h_samp: u8,
    pub v_samp: u8,
    pub quant_tbl_id: u8,
    pub dc_tbl_id: u8,
    pub ac_tbl_id: u8,
}

impl Default for ComponentInfo {
    fn default() -> Self {
        Self {
            id: 0,
            h_samp: 1,
            v_samp: 1,
            quant_tbl_id: 0,
            dc_tbl_id: 0,
            ac_tbl_id: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JpegHeader {
    pub width: usize,
    pub height: usize,
    pub precision: u8,
    pub components: Vec<ComponentInfo>,
    pub max_h_samp: u8,
    pub max_v_samp: u8,
    pub dqt: [[u16; 64]; 4],
    pub has_dqt: [bool; 4],
    pub restart_interval: u16,
    pub scan_offset: usize,
}

impl JpegHeader {
    pub fn mcu_width(&self) -> usize {
        (self.max_h_samp as usize) * 8
    }

    pub fn mcu_height(&self) -> usize {
        (self.max_v_samp as usize) * 8
    }

    pub fn mcus_x(&self) -> usize {
        (self.width + self.mcu_width() - 1) / self.mcu_width()
    }

    pub fn mcus_y(&self) -> usize {
        (self.height + self.mcu_height() - 1) / self.mcu_height()
    }
}
