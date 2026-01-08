// Label support for spatial data (bboxes, keypoints, masks)
//
// This module provides zero-copy wrappers and format conversion utilities
// for spatial labels that accompany images during augmentation.

pub mod format;
pub mod bbox;
pub mod keypoint;

#[cfg(test)]
mod tests;

// Re-export public types
pub use format::{BBoxFormat, KeypointFormat};
pub use bbox::{BBoxArray, BBoxArrayOwned};
pub use keypoint::{KeypointArray, KeypointArrayOwned};
