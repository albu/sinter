// Tests for matrix fusion

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::core::FusableImage;

    // Mock MatrixOp for testing
    #[derive(Debug, Clone, Copy)]
    struct TestMatrix {
        matrix: [[f32; 3]; 3],
    }

    impl MatrixOp for TestMatrix {
        fn get_matrix(&self) -> [[f32; 3]; 3] {
            self.matrix
        }
    }

    #[test]
    fn test_identity_matrix() {
        let identity = [[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]];

        let mut data = vec![100u8, 150u8, 200u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        apply_matrix(&mut img, &identity);

        assert_eq!(img.data[0], 100);
        assert_eq!(img.data[1], 150);
        assert_eq!(img.data[2], 200);
    }

    #[test]
    fn test_grayscale_matrix() {
        // Standard RGB to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
        let grayscale = [[0.299, 0.587, 0.114],
                         [0.299, 0.587, 0.114],
                         [0.299, 0.587, 0.114]];

        let mut data = vec![255u8, 0u8, 0u8];  // Pure red
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        apply_matrix(&mut img, &grayscale);

        // Y ≈ 76 for pure red (255 * 0.299)
        let expected = (255.0 * 0.299) as u8;
        assert_eq!(img.data[0], expected);
        assert_eq!(img.data[1], expected);
        assert_eq!(img.data[2], expected);
    }

    #[test]
    fn test_sepia_matrix() {
        // Standard sepia matrix
        let sepia = [[0.393, 0.769, 0.189],
                     [0.349, 0.686, 0.168],
                     [0.272, 0.534, 0.131]];

        let mut data = vec![255u8, 0u8, 0u8];  // Pure red
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        apply_matrix(&mut img, &sepia);

        // R' = 255 * 0.393 ≈ 100
        assert!((img.data[0] as i32 - 100).abs() <= 1);
        // G' = 255 * 0.349 ≈ 89
        assert!((img.data[1] as i32 - 89).abs() <= 1);
        // B' = 255 * 0.272 ≈ 69
        assert!((img.data[2] as i32 - 69).abs() <= 1);
    }

    #[test]
    fn test_matrix_composition() {
        let identity = TestMatrix {
            matrix: [[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
        };

        let double_red = TestMatrix {
            matrix: [[2.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
        };

        let half_red = TestMatrix {
            matrix: [[0.5, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
        };

        // Compose: double_red → half_red → identity
        // Should give us back: 2.0 * 0.5 = 1.0 for red channel
        let ops: &[&dyn MatrixOp] = &[&identity, &half_red, &double_red];
        let composed = compose_matrices(ops);

        // Red component should be 1.0 (2.0 * 0.5 * 1.0)
        assert!((composed[0][0] - 1.0).abs() < 0.001);
        assert_eq!(composed[0][1], 0.0);
        assert_eq!(composed[0][2], 0.0);
    }

    #[test]
    fn test_compose_single_matrix() {
        let sepia = TestMatrix {
            matrix: [[0.393, 0.769, 0.189],
                     [0.349, 0.686, 0.168],
                     [0.272, 0.534, 0.131]],
        };

        let ops: &[&dyn MatrixOp] = &[&sepia];
        let composed = compose_matrices(ops);

        // Should equal the original sepia matrix
        for i in 0..3 {
            for j in 0..3 {
                assert!((composed[i][j] - sepia.matrix[i][j]).abs() < 0.001);
            }
        }
    }

    #[test]
    fn test_executor_apply() {
        let sepia = TestMatrix {
            matrix: [[0.393, 0.769, 0.189],
                     [0.349, 0.686, 0.168],
                     [0.272, 0.534, 0.131]],
        };

        let mut data = vec![128u8, 128u8, 128u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        MatrixExecutor::apply(&mut img, &sepia.matrix);

        // Sepia of gray (128,128,128) should give sepia tone
        // R' = 128*0.393 + 128*0.769 + 128*0.189 = 128*(0.393+0.769+0.189) = 128*1.351 = 173
        // G' = 128*0.349 + 128*0.686 + 128*0.168 = 128*(0.349+0.686+0.168) = 128*1.203 = 154
        // B' = 128*0.272 + 128*0.534 + 128*0.131 = 128*(0.272+0.534+0.131) = 128*0.937 = 120
        let r = img.data[0];
        let g = img.data[1];
        let b = img.data[2];

        // Verify values are in expected sepia range (within 5 of expected)
        assert!((r as i32 - 173).abs() <= 2);
        assert!((g as i32 - 154).abs() <= 2);
        assert!((b as i32 - 120).abs() <= 2);
    }

    #[test]
    fn test_executor_fused_single() {
        let sepia = TestMatrix {
            matrix: [[0.393, 0.769, 0.189],
                     [0.349, 0.686, 0.168],
                     [0.272, 0.534, 0.131]],
        };

        let mut data = vec![100u8, 150u8, 200u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let ops: &[&dyn MatrixOp] = &[&sepia];
        MatrixExecutor::execute_fused(&mut img, ops);

        // Should produce sepia result
        let r = img.data[0];
        let g = img.data[1];
        let b = img.data[2];

        // Values should be transformed (not equal to original)
        assert_ne!(r, 100);
        assert_ne!(g, 150);
        assert_ne!(b, 200);
    }

    #[test]
    fn test_executor_fused_multiple() {
        let op1 = TestMatrix {
            matrix: [[1.5, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
        };

        let op2 = TestMatrix {
            matrix: [[0.5, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
        };

        let mut data = vec![100u8, 150u8, 200u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        // op1 then op2: 1.5 * 0.5 = 0.75 for red channel
        let ops: &[&dyn MatrixOp] = &[&op1, &op2];
        MatrixExecutor::execute_fused(&mut img, ops);

        // Red should be 100 * 1.5 * 0.5 = 75
        assert!((img.data[0] as i32 - 75).abs() <= 1);
        // Green and blue should be unchanged (1.0 * 1.0 = 1.0)
        assert_eq!(img.data[1], 150);
        assert_eq!(img.data[2], 200);
    }

    #[test]
    fn test_clamping() {
        // Matrix that produces values > 255
        let overflow = [[2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 2.0]];

        let mut data = vec![200u8, 200u8, 200u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        apply_matrix(&mut img, &overflow);

        // All values should be clamped to 255
        assert_eq!(img.data[0], 255);
        assert_eq!(img.data[1], 255);
        assert_eq!(img.data[2], 255);
    }

    #[test]
    fn test_clamping_negative() {
        // Matrix that produces negative values
        let negative = [[-0.5, 0.0, 0.0],
                        [0.0, -0.5, 0.0],
                        [0.0, 0.0, -0.5]];

        let mut data = vec![100u8, 100u8, 100u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        apply_matrix(&mut img, &negative);

        // All values should be clamped to 0
        assert_eq!(img.data[0], 0);
        assert_eq!(img.data[1], 0);
        assert_eq!(img.data[2], 0);
    }
}
