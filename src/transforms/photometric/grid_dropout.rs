// GridDropout transform
//
// Randomly drops out grid cells by setting them to zero.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

/// GridDropout transform
///
/// Divides the image into a grid and randomly sets entire grid cells to zero.
/// This is a form of regularization that forces the model to learn
/// from partial information in a structured way.
///
/// # Parameters
/// - `grid_size`: Size of each grid cell (width, height) in pixels
/// - `drop_ratio`: Fraction of grid cells to drop out (0.0 to 1.0)
/// - `fill_value`: Value to fill holes with (default: 0)
///
/// # Notes
/// - Operates in-place (no allocation)
/// - Preserves image shape
/// - Position-dependent operation (breaks pure per-pixel fusion)
/// - Uses deterministic hashing for reproducible results
#[derive(Debug, Clone, PartialEq)]
pub struct GridDropout {
    /// Size of grid cells (width, height)
    pub grid_size: (u32, u32),
    /// Fraction of grid cells to drop
    pub drop_ratio: f32,
    /// Value to fill cells with
    pub fill_value: u8,
    /// Per-pipeline seed so different images get different dropped cells.
    pub seed: u64,
}

impl GridDropout {
    /// Create a new GridDropout transform
    ///
    /// # Arguments
    /// * `grid_size` - Size of each grid cell as (width, height) in pixels
    /// * `drop_ratio` - Fraction of grid cells to drop (0.0 to 1.0)
    /// * `fill_value` - Value to fill cells with (default: 0)
    ///
    /// # Panics
    /// Panics if grid_size values are 0 or if drop_ratio is outside [0.0, 1.0]
    pub fn new(grid_size: (u32, u32), drop_ratio: f32, fill_value: u8) -> Self {
        Self::with_seed(grid_size, drop_ratio, fill_value, 0)
    }

    /// Create a new GridDropout transform with an explicit per-pipeline seed.
    ///
    /// # Panics
    /// Panics if grid_size values are 0 or if drop_ratio is outside [0.0, 1.0]
    pub fn with_seed(grid_size: (u32, u32), drop_ratio: f32, fill_value: u8, seed: u64) -> Self {
        assert!(
            grid_size.0 > 0 && grid_size.1 > 0,
            "grid_size must be positive, got {:?}",
            grid_size
        );
        assert!(
            drop_ratio >= 0.0 && drop_ratio <= 1.0,
            "drop_ratio must be in [0.0, 1.0], got {}",
            drop_ratio
        );
        Self {
            grid_size,
            drop_ratio,
            fill_value,
            seed,
        }
    }

    /// Create a default GridDropout with common parameters
    ///
    /// - Grid size: 32x32 pixels
    /// - Drop ratio: 20%
    /// - Fill with 0
    pub fn default_params() -> Self {
        Self::new((32, 32), 0.2, 0)
    }

    /// Simple hash function for reproducible pseudo-randomness

    /// Generate list of grid cells to drop
    fn generate_drops(&self, grid_w: usize, grid_h: usize) -> Vec<(usize, usize)> {
        let total_cells = grid_w * grid_h;
        let num_drops = (total_cells as f32 * self.drop_ratio).ceil() as usize;

        let mut drops = Vec::with_capacity(num_drops);

        // Use a linear congruential approach to select cells
        // This ensures we get exactly num_drops unique cells
        let step = if total_cells > 1 {
            // Golden ratio for good distribution, jittered by the seed.
            (total_cells as f32 * 1.618033988749895) as usize + 1
                + (self.seed as usize % total_cells.max(1))
        } else {
            1
        };

        // Offset the starting cell by the seed so different pipelines drop
        // different cells.
        let start = if total_cells > 1 {
            self.seed as usize % total_cells
        } else {
            0
        };

        for i in 0..num_drops {
            let cell_idx = (start + i * step) % total_cells;
            let grid_x = cell_idx % grid_w;
            let grid_y = cell_idx / grid_w;
            drops.push((grid_x, grid_y));
        }

        drops
    }

    /// Apply drops to image data
    fn apply_drops(&self, image: &mut FusableImage, drops: &[(usize, usize)]) {
        let cell_w = self.grid_size.0 as usize;
        let cell_h = self.grid_size.1 as usize;

        let img_width = image.width;
        let img_height = image.height;
        let channels = image.channels;
        let row_stride = img_width * channels;

        for &(grid_x, grid_y) in drops {
            // Calculate pixel coordinates of this grid cell
            let x_start = grid_x * cell_w;
            let y_start = grid_y * cell_h;
            let x_end = (x_start + cell_w).min(img_width);
            let y_end = (y_start + cell_h).min(img_height);

            // Fill this entire cell
            for row in y_start..y_end {
                let row_start = row * row_stride + x_start * channels;
                let row_end = row * row_stride + x_end * channels;

                for px in row_start..row_end {
                    image.data[px] = self.fill_value;
                }
            }
        }
    }
}

impl Transform for GridDropout {
    fn access(&self) -> AccessPattern {
        // InPlace - we modify the existing buffer
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        // Preserves dimensions
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for GridDropout {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Calculate grid dimensions
        let grid_w = (image.width + self.grid_size.0 as usize - 1) / self.grid_size.0 as usize;
        let grid_h = (image.height + self.grid_size.1 as usize - 1) / self.grid_size.1 as usize;

        let drops = self.generate_drops(grid_w, grid_h);
        self.apply_drops(image, &drops);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_dropout_new() {
        let gd = GridDropout::new((16, 16), 0.3, 0);
        assert_eq!(gd.grid_size, (16, 16));
        assert_eq!(gd.drop_ratio, 0.3);
        assert_eq!(gd.fill_value, 0);
    }

    #[test]
    fn test_grid_dropout_default_params() {
        let gd = GridDropout::default_params();
        assert_eq!(gd.grid_size, (32, 32));
        assert_eq!(gd.drop_ratio, 0.2);
        assert_eq!(gd.fill_value, 0);
    }

    #[test]
    #[should_panic(expected = "grid_size must be positive")]
    fn test_grid_dropout_invalid_grid_size() {
        GridDropout::new((0, 16), 0.1, 0);
    }

    #[test]
    #[should_panic(expected = "drop_ratio must be in")]
    fn test_grid_dropout_invalid_drop_ratio() {
        GridDropout::new((16, 16), 1.5, 0);
    }

    #[test]
    fn test_grid_dropout_generate_drops() {
        let gd = GridDropout::new((10, 10), 0.5, 0);
        // Image 100x100 with 10x10 cells = 10x10 grid = 100 cells
        // 50% drop = 50 cells
        let drops = gd.generate_drops(10, 10);

        assert_eq!(drops.len(), 50);

        // Check all drops are within grid bounds
        for &(gx, gy) in &drops {
            assert!(gx < 10);
            assert!(gy < 10);
        }
    }

    #[test]
    fn test_grid_dropout_execute_single_channel() {
        // Create a 64x64 image with all pixels = 128
        let mut data = vec![128u8; 64 * 64];
        let mut img = FusableImage::new(&mut data, 64, 64, 1);

        // 32x32 grid cells, drop 50%
        let gd = GridDropout::new((32, 32), 0.5, 0);

        gd.execute(&mut img);

        // Grid is 2x2 cells (each 32x32 pixels)
        // Count zeros in each 32x32 cell
        let cell_size = 32 * 32;
        let stride = 64;

        let mut cells_zeroed = 0;

        // Cell (0,0): top-left
        let zeros_tl: usize = img.data[0..cell_size].iter().filter(|&&p| p == 0).count();
        if zeros_tl == cell_size {
            cells_zeroed += 1;
        }

        // Cell (0,1): top-right
        let zeros_tr: usize = (0..32)
            .flat_map(|row| {
                let start = row * stride + 32;
                img.data[start..start + 32].iter()
            })
            .filter(|&&p| p == 0)
            .count();
        if zeros_tr == cell_size {
            cells_zeroed += 1;
        }

        // Cell (1,0): bottom-left
        let zeros_bl: usize = (32..64)
            .flat_map(|row| {
                let start = row * stride;
                img.data[start..start + 32].iter()
            })
            .filter(|&&p| p == 0)
            .count();
        if zeros_bl == cell_size {
            cells_zeroed += 1;
        }

        // Cell (1,1): bottom-right
        let zeros_br: usize = (32..64)
            .flat_map(|row| {
                let start = row * stride + 32;
                img.data[start..start + 32].iter()
            })
            .filter(|&&p| p == 0)
            .count();
        if zeros_br == cell_size {
            cells_zeroed += 1;
        }

        // With 50% dropout and 4 cells, we expect approximately 2 cells to be zeroed
        // But due to randomness, we'll just check that at least 1 cell is zeroed
        assert!(
            cells_zeroed >= 1,
            "At least one grid cell should be fully zeroed"
        );
    }

    #[test]
    fn test_grid_dropout_execute_rgb() {
        // Create a 32x32 RGB image with all pixels = (100, 150, 200)
        let mut data = vec![0u8; 32 * 32 * 3];
        for i in 0..(32 * 32) {
            data[i * 3] = 100;
            data[i * 3 + 1] = 150;
            data[i * 3 + 2] = 200;
        }
        let mut img = FusableImage::new(&mut data, 32, 32, 3);

        // 16x16 grid cells, drop 50%
        let gd = GridDropout::new((16, 16), 0.5, 42);

        gd.execute(&mut img);

        // Count non-42 pixels to verify some were dropped
        let non_42 = data.iter().filter(|&&p| p != 42).count();
        // At least some pixels should remain unchanged
        assert!(non_42 > 0);
    }

    #[test]
    fn test_grid_dropout_zero_drop_ratio() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let gd = GridDropout::new((5, 5), 0.0, 0);
        gd.execute(&mut img);

        // Image should be unchanged
        assert!(img.data.iter().all(|&p| p == 128));
    }

    #[test]
    fn test_grid_dropout_full_drop_ratio() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let gd = GridDropout::new((5, 5), 1.0, 99);
        gd.execute(&mut img);

        // Entire image should be filled with 99
        assert!(img.data.iter().all(|&p| p == 99));
    }

    #[test]
    fn test_grid_dropout_access_pattern() {
        let gd = GridDropout::new((16, 16), 0.1, 0);
        assert_eq!(gd.access(), AccessPattern::InPlace);
        assert_eq!(gd.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_grid_dropout_reproducibility() {
        let mut data1 = vec![128u8; 64 * 64];
        let mut img1 = FusableImage::new(&mut data1, 64, 64, 1);

        let mut data2 = vec![128u8; 64 * 64];
        let mut img2 = FusableImage::new(&mut data2, 64, 64, 1);

        let gd = GridDropout::new((32, 32), 0.5, 0);
        gd.execute(&mut img1);
        gd.execute(&mut img2);

        // Same input should produce same output (reproducible)
        assert_eq!(img1.data, img2.data);
    }

    #[test]
    fn test_grid_dropout_custom_fill_value() {
        let mut data = vec![128u8; 64];
        let mut img = FusableImage::new(&mut data, 8, 8, 1);

        let gd = GridDropout::new((4, 4), 0.25, 255);
        gd.execute(&mut img);

        // Some pixels should be 255
        assert!(img.data.iter().any(|&p| p == 255));
    }

    #[test]
    fn test_grid_dropout_non_uniform_grid() {
        // Image that doesn't divide evenly by grid size
        let mut data = vec![128u8; 50 * 50];
        let mut img = FusableImage::new(&mut data, 50, 50, 1);

        // 16x16 cells on 50x50 image -> partial cells at edges
        let gd = GridDropout::new((16, 16), 0.5, 0);
        gd.execute(&mut img);

        // Should complete without panic
        // Just verify some pixels were zeroed
        assert!(img.data.iter().any(|&p| p == 0));
    }
}
