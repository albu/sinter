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
    fn test_separable_3x3_median_network() {
        // Given 3 sorted columns C0=(a0<=a1<=a2), C1=(b0<=b1<=b2), C2=(c0<=c1<=c2)
        // Find median of all 9 elements using 7 comparators!
        fn median_3x3_separable(
            a0: u8, a1: u8, a2: u8,
            b0: u8, b1: u8, b2: u8,
            c0: u8, c1: u8, c2: u8,
        ) -> u8 {
            // 1. max of 3 minimums (2 comps)
            let max_min = a0.max(b0).max(c0);
            // 2. min of 3 maximums (2 comps)
            let min_max = a2.min(b2).min(c2);
            // 3. Middle elements: a1, b1, c1
            // We now have 5 elements: [max_min, a1, b1, c1, min_max]
            // Let's find median of these 5 elements:
            let mut v0 = max_min;
            let mut v1 = a1;
            let mut v2 = b1;
            let mut v3 = c1;
            let mut v4 = min_max;

            // 5-element median selection network (6 comps)
            // v0-v1, v3-v4, v0-v3, v1-v4, v1-v2, v2-v3, v1-v2
            let min = v0.min(v1); let max = v0.max(v1); v0 = min; v1 = max;
            let min = v3.min(v4); let max = v3.max(v4); v3 = min; v4 = max;
            let min = v0.min(v3); let max = v0.max(v3); v0 = min; v3 = max;
            let min = v1.min(v4); let max = v1.max(v4); v1 = min; v4 = max;
            let min = v1.min(v2); let max = v1.max(v2); v1 = min; v2 = max;
            let min = v2.min(v3); let max = v2.max(v3); v2 = min; v3 = max;
            let min = v1.min(v2); let max = v1.max(v2); v1 = min; v2 = max;

            v2
        }

        // Verify with 0-1 lemma on all 512 binary states
        for mask in 0..512 {
            let mut raw = [0u8; 9];
            for i in 0..9 {
                raw[i] = ((mask >> i) & 1) as u8;
            }
            let mut col0 = [raw[0], raw[3], raw[6]]; col0.sort();
            let mut col1 = [raw[1], raw[4], raw[7]]; col1.sort();
            let mut col2 = [raw[2], raw[5], raw[8]]; col2.sort();

            let mut sorted = raw;
            sorted.sort();
            let expected_median = sorted[4];

            let actual = median_3x3_separable(
                col0[0], col0[1], col0[2],
                col1[0], col1[1], col1[2],
                col2[0], col2[1], col2[2],
            );

            assert_eq!(actual, expected_median, "Failed for mask {:09b}", mask);
        }
        println!("Separable 3x3 median network mathematically proven for 100% of states!");
    }

    #[test]
    fn test_column_sorted_median25_network() {
        // Generate all 7776 sorted 5-column states (each column is 0..=5 ones)
        let mut states: Vec<[u8; 25]> = Vec::new();
        for c0 in 0..=5 {
            for c1 in 0..=5 {
                for c2 in 0..=5 {
                    for c3 in 0..=5 {
                        for c4 in 0..=5 {
                            let mut st = [0u8; 25];
                            for i in 0..c0 { st[i] = 1; }
                            for i in 0..c1 { st[5 + i] = 1; }
                            for i in 0..c2 { st[10 + i] = 1; }
                            for i in 0..c3 { st[15 + i] = 1; }
                            for i in 0..c4 { st[20 + i] = 1; }
                            states.push(st);
                        }
                    }
                }
            }
        }
        assert_eq!(states.len(), 7776);

        // Pre-compute expected median for each state
        let mut targets = Vec::with_capacity(7776);
        for st in &states {
            let mut sorted = *st;
            sorted.sort();
            targets.push(sorted[12]);
        }

        // Generate Batcher odd-even mergesort network for N=32
        let mut network: Vec<(usize, usize)> = Vec::new();
        let mut p = 1;
        while p < 32 {
            let mut k = p;
            while k >= 1 {
                let mut j = k % p;
                while j <= 31 - k {
                    let max_i = k.min(31 - k - j);
                    for i in 0..=max_i {
                        if (i + j) / (p * 2) == (i + j + k) / (p * 2) {
                            network.push((i + j, i + j + k));
                        }
                    }
                    j += 2 * k;
                }
                k /= 2;
            }
            p *= 2;
        }

        // Pad each state to 32 elements (3 zeros at start, 4 ones at end)
        // Expected median of 25 items is at index 3 + 12 = 15
        let mut correct = 0;
        for (st, &expected) in states.iter().zip(targets.iter()) {
            let mut arr = [0u8; 32];
            arr[..3].fill(0);
            arr[3..28].copy_from_slice(st);
            arr[28..32].fill(1);

            for &(a, b) in &network {
                let min = arr[a].min(arr[b]);
                let max = arr[a].max(arr[b]);
                arr[a] = min;
                arr[b] = max;
            }

            if arr[15] == expected {
                correct += 1;
            }
        }

        assert_eq!(correct, 7776, "Batcher 32 MUST pass all 7776 states!");
        println!("Batcher 32 passed 100% of states!");
    }
}

// Historical note (concluded negative): a search for a minimal comparator
// network computing the median of 25 from five sorted columns was run and
// conclusively closed. Greedy pruning of a verified 204-comparator seed
// reached 108; annealing reached 93 — above the pre-registered 65 kill
// criterion — and the resulting network failed held-out fuzzing despite
// passing the entire binary-sorted domain (the 0/1 lemma does not hold for
// sorted-column-restricted byte domains). Sliding-column 5x5 median stays
// dead; the O(1) histogram algorithm is the only path to cv2-level 5x5.
// The full search harness is preserved in git history (commit 8d84b38).
