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

/// Search for a minimal comparator network that computes the median of 25
/// when the inputs arrive as five sorted 5-element columns. This is the
/// combine half of a sliding-column 5x5 median: sort5 per column is paid
/// once per window slide, so the combine only needs to be exact on
/// already-sorted columns.
///
/// Wire layout: wire `5*c + j` holds element `j` (0 = smallest) of column
/// `c`. Verification set: every binary input whose columns are sorted
/// (6 shapes per column => 6^5 = 7776 cases). The median of a binary input
/// is 1 iff popcount >= 13. Candidate networks are additionally fuzzed on
/// millions of random sorted-column byte inputs before being trusted.
///
/// Run with: cargo test --release search_median25_sorted_combine -- --ignored --nocapture
#[cfg(test)]
mod combine_search {
    type Net = Vec<(u8, u8)>;

    fn apply(net: &Net, wires: &mut [u8; 25]) {
        for &(a, b) in net {
            let (x, y) = (wires[a as usize], wires[b as usize]);
            if x > y {
                wires[a as usize] = y;
                wires[b as usize] = x;
            }
        }
    }

    /// Bitonic sort on 32 wires, filtered to the first 25. Wires 25..31 act
    /// as +Infinity padding: any comparator touching them is the identity
    /// (min(real, +inf) = real), so deleting those comparators leaves a
    /// correct sorting network for the 25 real wires. Generation is trivially
    /// correct for power-of-two sizes, unlike hand-rolled Batcher variants.
    fn seed_sort25() -> Net {
        // Standard Batcher odd-even mergesort for n=32 (power-of-two correct),
        // filtered to the 25 real wires. Wires 25..31 act as +Infinity pads:
        // every comparator touching them has the real wire at the lower index,
        // so min(x, +inf) = x keeps it identity — filtering preserves sorting.
        let mut network: Net = Vec::new();
        let mut p = 1usize;
        while p < 32 {
            let mut k = p;
            while k >= 1 {
                let mut j = k % p;
                while j <= 31 - k {
                    let max_i = k.min(31 - k - j);
                    for i in 0..=max_i {
                        if (i + j) / (p * 2) == (i + j + k) / (p * 2) {
                            network.push(((i + j) as u8, (i + j + k) as u8));
                        }
                    }
                    j += 2 * k;
                }
                k /= 2;
            }
            p *= 2;
        }
        network.retain(|&(a, b)| (a as usize) < 25 && (b as usize) < 25);
        network
    }

    /// All 6^5 binary inputs with sorted columns.
    fn binary_cases() -> Vec<([u8; 25], u8)> {
        let mut cases = Vec::with_capacity(7_776);
        for &counts in [
            [0u32, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4], [5, 5, 5, 5, 5],
        ]
        .iter()
        .take(0)
        {}
        for c0 in 0..=5i32 {
            for c1 in 0..=5i32 {
                for c2 in 0..=5i32 {
                    for c3 in 0..=5i32 {
                        for c4 in 0..=5i32 {
                            let counts = [c0, c1, c2, c3, c4];
                            let mut input = [0u8; 25];
                            let mut ones = 0i32;
                            for (c, &n) in counts.iter().enumerate() {
                                ones += n;
                                for j in 0..n as usize {
                                    input[5 * c + j] = 1;
                                }
                            }
                            let _ = (5 - ones.unsigned_abs()) as usize;
                            cases.push((input, u8::from(ones >= 13)));
                        }
                    }
                }
            }
        }
        cases
    }

    /// Fixed adversarial byte corpus: random sorted-column inputs over small
    /// alphabets (ties are where median networks break) plus full random.
    fn byte_corpus(seed: u64, n: usize) -> Vec<([u8; 25], u8)> {
        let mut rng = seed | 1;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut cases = Vec::with_capacity(n);
        for i in 0..n {
            let alphabet = match i % 4 {
                0 => 2u64,
                1 => 3,
                2 => 8,
                _ => 256,
            };
            let mut input = [0u8; 25];
            for c in 0..5 {
                let mut v = [0u8; 5];
                for val in v.iter_mut() {
                    *val = ((next() % alphabet.max(2)) & 0xFF) as u8;
                }
                v.sort_unstable();
                input[5 * c..5 * c + 5].copy_from_slice(&v);
            }
            let mut sorted_all = input;
            sorted_all.sort_unstable();
            cases.push((input, sorted_all[12]));
        }
        cases
    }

    fn is_exact(net: &Net, cases: &[([u8; 25], u8)]) -> bool {
        let mut wires = [0u8; 25];
        for (input, expected) in cases {
            wires = *input;
            apply(net, &mut wires);
            if wires[12] != *expected {
                return false;
            }
        }
        true
    }

    /// CONCLUDED (negative): greedy pruning 204 -> 108 and 257K annealing
    /// iterations reach only 93 comparators — above the 65 kill criterion —
    /// and the 93-net that passes all 7,776 binary-sorted cases + a 4K byte
    /// corpus still fails real byte inputs (overfits the corpus). Cheap
    /// exhaustive verification does not exist for the sorted-column byte
    /// domain, so a searched combine cannot be trusted. 5x5 stays on the
    /// sortnet; the only path to cv2-level 5x5 is the O(1) histogram
    /// algorithm (rewrite-class). Kept for the record; fuzz reports instead
    /// of asserting.
    #[test]
    #[ignore = "concluded negative: cargo test --release search_median25_sorted_combine -- --ignored -- --nocapture"]
    fn search_median25_sorted_combine() {
        let binary = binary_cases();
        let byte_cases = byte_corpus(0xDEADBEEF, 4_000);
        let mut fitness_cases = binary;
        fitness_cases.extend_from_slice(&byte_cases);
        println!("fitness cases: {}", fitness_cases.len());

        // Seed: verify the construction is a true sort on random vectors.
        let seed = seed_sort25();
        println!("seed sort25: {} comparators", seed.len());
        for k in 0..1000u64 {
            let mut v = [0u8; 25];
            let mut r = 0x9E3779B97F4A7C15u64 ^ (k.wrapping_mul(0xBF58476D1CE4E5B9));
            for b in v.iter_mut() {
                r ^= r << 13;
                r ^= r >> 7;
                r ^= r << 17;
                *b = (r >> 24) as u8;
            }
            let mut sorted_v = v;
            sorted_v.sort_unstable();
            apply(&seed, &mut v);
            assert_eq!(v, sorted_v, "seed network is not a correct sort");
        }

        // ---- Strategy 1: greedy prune of the correct seed ----
        let mut net = seed;
        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            while i < net.len() {
                let candidate: Net = net
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &p)| p)
                    .collect();
                if is_exact(&candidate, &fitness_cases) {
                    net = candidate;
                    changed = true;
                } else {
                    i += 1;
                }
            }
        }
        println!("greedy prune: {} comparators", net.len());

        // ---- Strategy 2: annealing ----
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut best = net;
        let mut cur = best.clone();
        let iterations = 300_000usize;
        for it in 0..iterations {
            let temperature = 8 - 8 * it / iterations;
            let mut trial = cur.clone();
            match (next() as usize) % 3 {
                0 if !trial.is_empty() => {
                    trial.remove((next() as usize) % trial.len());
                }
                1 => {
                    let i = (next() as usize) % 25;
                    let mut j = (next() as usize) % 25;
                    while j == i {
                        j = (next() as usize) % 25;
                    }
                    trial.push((i as u8, j as u8));
                }
                _ if !trial.is_empty() => {
                    let i = (next() as usize) % trial.len();
                    let a = (next() as usize) % 25;
                    let b = (next() as usize) % 25;
                    if a != b {
                        trial[i] = (a as u8, b as u8);
                    }
                }
                _ => continue,
            }
            if !is_exact(&trial, &fitness_cases) {
                continue;
            }
            let dl = trial.len() as isize - cur.len() as isize;
            let keep = dl <= 0 || ((next() as usize) % 1000) < temperature * 40;
            if keep {
                cur = trial;
                if cur.len() < best.len() {
                    best = cur.clone();
                    println!("[{:6}] new best: {} comparators", it, best.len());
                }
            }
        }
        println!("FINAL: {} comparators", best.len());
        println!("{:?}", best);

        // ---- Fuzz the winner: report accuracy instead of asserting ----
        let mut checked = 0usize;
        let mut wrong = 0usize;
        for batch in 0..20 {
            let corpus = byte_corpus(0xC0FFEE + batch as u64, 1_000_000);
            for (input, expected) in &corpus {
                let mut wires = *input;
                apply(&best, &mut wires);
                if wires[12] != *expected {
                    wrong += 1;
                }
            }
            checked += corpus.len();
        }
        println!(
            "fuzz: {} cases, {} wrong ({:.4}%) -> {}",
            checked,
            wrong,
            wrong as f64 / checked as f64 * 100.0,
            if wrong == 0 { "SOUND (statistically)" } else { "UNSOUND — overfits the fitness corpus" }
        );
    }
}
