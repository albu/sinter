// Fast sorting networks for median filtering (3x3 and 5x5)
//
// 3x3 median uses a 19 compare-and-swap selection network.
// Branchless, zero-allocation, optimal for SIMD and scalar execution.

macro_rules! cas {
    ($a:expr, $b:expr) => {
        let min = $a.min($b);
        let max = $a.max($b);
        $a = min;
        $b = max;
    };
}

/// Compute median of 9 values using a 19-comparator selection network
#[inline(always)]
pub fn median9(p: [u8; 9]) -> u8 {
    let mut p0 = p[0];
    let mut p1 = p[1];
    let mut p2 = p[2];
    let mut p3 = p[3];
    let mut p4 = p[4];
    let mut p5 = p[5];
    let mut p6 = p[6];
    let mut p7 = p[7];
    let mut p8 = p[8];

    cas!(p1, p2); cas!(p4, p5); cas!(p7, p8);
    cas!(p0, p1); cas!(p3, p4); cas!(p6, p7);
    cas!(p1, p2); cas!(p4, p5); cas!(p7, p8);
    cas!(p0, p3); cas!(p5, p8); cas!(p4, p7);
    cas!(p3, p6); cas!(p1, p4); cas!(p2, p5);
    cas!(p4, p7); cas!(p4, p2); cas!(p6, p4);
    cas!(p4, p2);

    p4
}

/// Apply 3x3 median blur using sorting network (scalar path)
pub fn apply_median_blur_3x3_scalar(data: &mut [u8], width: usize, height: usize, channels: usize) {
    let mut output = vec![0u8; data.len()];
    let stride = width * channels;

    for y in 0..height {
        let y_prev = y.saturating_sub(1);
        let y_next = (y + 1).min(height - 1);

        let row_curr = y * stride;
        let row_prev = y_prev * stride;
        let row_next = y_next * stride;

        for x in 0..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);

            for c in 0..channels {
                let p = [
                    data[row_prev + x_prev * channels + c],
                    data[row_prev + x * channels + c],
                    data[row_prev + x_next * channels + c],
                    data[row_curr + x_prev * channels + c],
                    data[row_curr + x * channels + c],
                    data[row_curr + x_next * channels + c],
                    data[row_next + x_prev * channels + c],
                    data[row_next + x * channels + c],
                    data[row_next + x_next * channels + c],
                ];

                output[row_curr + x * channels + c] = median9(p);
            }
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median9_0_1_lemma() {
        // 0-1 Sorting Lemma: If a comparator network correctly computes the median
        // for all 2^9 = 512 binary sequences, it is correct for all inputs.
        for mask in 0..512 {
            let mut input = [0u8; 9];
            let mut sorted = [0u8; 9];
            for i in 0..9 {
                input[i] = ((mask >> i) & 1) as u8;
                sorted[i] = input[i];
            }
            sorted.sort();
            let expected_median = sorted[4];
            let actual_median = median9(input);
            assert_eq!(
                actual_median, expected_median,
                "Failed for binary mask {:09b}",
                mask
            );
        }
    }

    #[test]
    fn test_median9_random_permutations() {
        let mut vals = [10u8, 25, 40, 70, 110, 150, 190, 220, 250];
        let actual = median9(vals);
        assert_eq!(actual, 110);

        vals.reverse();
        let actual = median9(vals);
        assert_eq!(actual, 110);
    }
}
