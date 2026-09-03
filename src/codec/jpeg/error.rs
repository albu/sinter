use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JpegError {
    InvalidMarker(u16),
    UnexpectedEof,
    UnsupportedPrecision(u8),
    UnsupportedComponentCount(usize),
    UnsupportedSubsampling,
    UnsupportedEncoding(String),
    MissingSOF,
    MissingSOS,
    MissingDQT,
    MissingDHT,
    CorruptedBitstream(String),
    InvalidDimensions(usize, usize),
    CropOutOfBounds,
}

impl fmt::Display for JpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMarker(m) => write!(f, "Invalid JPEG marker: 0x{:04X}", m),
            Self::UnexpectedEof => write!(f, "Unexpected end of JPEG stream"),
            Self::UnsupportedPrecision(p) => write!(f, "Unsupported sample precision: {} bits (only 8-bit supported)", p),
            Self::UnsupportedComponentCount(c) => write!(f, "Unsupported component count: {} (expected 1 or 3)", c),
            Self::UnsupportedSubsampling => write!(f, "Unsupported chroma subsampling ratio"),
            Self::UnsupportedEncoding(s) => write!(f, "Unsupported JPEG encoding mode: {}", s),
            Self::MissingSOF => write!(f, "Missing SOF (Start of Frame) marker in JPEG header"),
            Self::MissingSOS => write!(f, "Missing SOS (Start of Scan) marker in JPEG header"),
            Self::MissingDQT => write!(f, "Quantization table referenced by component was not defined"),
            Self::MissingDHT => write!(f, "Huffman table referenced by component was not defined"),
            Self::CorruptedBitstream(msg) => write!(f, "Corrupted JPEG bitstream: {}", msg),
            Self::InvalidDimensions(w, h) => write!(f, "Invalid image dimensions: {}x{}", w, h),
            Self::CropOutOfBounds => write!(f, "Requested crop region extends beyond image bounds"),
        }
    }
}

impl std::error::Error for JpegError {}
