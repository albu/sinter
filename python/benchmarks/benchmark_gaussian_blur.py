"""
GaussianBlur benchmark - comprehensive performance analysis.

Tests different image sizes x different kernel sizes to understand performance characteristics.
"""
import time
import numpy as np

try:
    import cv2
    cv2.setNumThreads(0)
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not installed")

from sinter import GaussianBlur as SinterGaussianBlur, Compose


WARMUP_RUNS = 3
BENCHMARK_RUNS = 30

# Test different image sizes
IMAGE_SIZES = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]

# Test different kernel sizes (multi-pass 7x7 strategy)
# 13x13 = 2 passes of 7x7, 21x21 = 3 passes of 7x7, 31x31 = 5 passes of 7x7
KERNEL_SIZES = [3, 5, 7, 13, 21, 31]

# Map kernel sizes to equivalent OpenCV sigma values
# These match the Pascal binomial kernels' effective sigma
SIGMA_MAP = {
    3: 0.85,   # [1,2,1] Pascal
    5: 1.08,   # [1,4,6,4,1] Pascal
    7: 1.28,   # [1,6,15,20,15,6,1] Pascal
    13: 1.70,  # 2×7×7 multi-pass (variance addition)
    21: 2.08,  # 3×7×7 multi-pass
    31: 2.68,  # 5×7×7 multi-pass
}


def benchmark_single(image_size, kernel_size):
    """Benchmark a single image size + kernel size combination."""
    height, width = image_size[:2]
    channels = image_size[2] if len(image_size) > 2 else 3

    img = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

    # Sinter GaussianBlur (our implementation)
    sinter_pipe = Compose([SinterGaussianBlur(kernel_size=kernel_size)])

    for _ in range(WARMUP_RUNS):
        _ = sinter_pipe.apply(img.copy())

    start = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        _ = sinter_pipe.apply(img.copy())
    sinter_time = (time.perf_counter() - start) / BENCHMARK_RUNS * 1000

    # OpenCV GaussianBlur (for comparison)
    cv_time = None
    if HAS_CV2:
        sigma = SIGMA_MAP.get(kernel_size)  # Must match Pascal kernel sigma

        for _ in range(WARMUP_RUNS):
            _ = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            _ = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        cv_time = (time.perf_counter() - start) / BENCHMARK_RUNS * 1000

    return cv_time, sinter_time


def run_benchmarks():
    """Run comprehensive benchmarks across image sizes and kernel sizes."""
    print("=" * 100)
    print("GAUSSIAN BLUR BENCHMARK - Multi-pass 7x7 Strategy")
    print("=" * 100)
    print(f"\nTesting {len(IMAGE_SIZES)} image sizes x {len(KERNEL_SIZES)} kernel sizes")
    print(f"Kernel sizes: {KERNEL_SIZES} (13x13=2×7×7, 21x21=3×7×7, 31x31=5×7×7)")
    print(f"Warmup runs: {WARMUP_RUNS}, Benchmark runs: {BENCHMARK_RUNS}")
    print()

    # Header row with kernel sizes
    header = f"{'Image Size':<15}"
    for ks in KERNEL_SIZES:
        header += f"  {ks}x{ks:<7}"
    print(header)
    print("-" * 100)

    # Results storage for summary
    results = {}

    for img_size in IMAGE_SIZES:
        h, w = img_size[0], img_size[1]
        pixels = h * w

        row = f"{h}x{w:<10}"
        results[f"{h}x{w}"] = {}

        for ks in KERNEL_SIZES:
            cv_time, sinter_time = benchmark_single((h, w, 3), ks)
            results[f"{h}x{w}"][ks] = (cv_time, sinter_time)

            row += f"  {sinter_time:>7.2f}m"

        print(row)

    # Print detailed comparison with OpenCV if available
    if HAS_CV2:
        print("\n" + "=" * 100)
        print("DETAILED COMPARISON vs OpenCV")
        print("=" * 100)
        print(f"{'Image Size':<15} {'Kernel':<10} {'OpenCV':>10} {'sinter':>10} {'Speedup':>10} {'Verdict':>8}")
        print("-" * 100)

        wins = 0
        total = 0

        for img_size in IMAGE_SIZES:
            h, w = img_size[0], img_size[1]
            for ks in KERNEL_SIZES:
                cv_time, sinter_time = results[f"{h}x{w}"][ks]
                speedup = cv_time / sinter_time
                verdict = "WIN" if speedup > 1 else "LOSS"
                marker = "⚡" if speedup > 1.5 else "✅" if speedup > 1 else "❌"

                if speedup > 1:
                    wins += 1
                total += 1

                print(f"{h}x{w:<10} {ks}x{ks:<6} {cv_time:>10.2f}m {sinter_time:>10.2f}m {speedup:>10.2f}x {marker} {verdict}")

        print(f"\nSummary: {wins}/{total} tests faster than OpenCV ({wins/total*100:.1f}%)")

    # Performance scaling analysis
    print("\n" + "=" * 100)
    print("PERFORMANCE SCALING ANALYSIS")
    print("=" * 100)

    # Analyze scaling with image size (for each kernel)
    print("\nImage Size Scaling (for each kernel size):")
    print("-" * 100)
    for ks in KERNEL_SIZES:
        print(f"\n{ks}x{ks} kernel:")
        for i in range(len(IMAGE_SIZES) - 1):
            h1, w1 = IMAGE_SIZES[i]
            h2, w2 = IMAGE_SIZES[i + 1]
            _, t1 = results[f"{h1}x{w1}"][ks]
            _, t2 = results[f"{h2}x{w2}"][ks]

            # Expected scaling (quadratic)
            expected_ratio = (h2 * w2) / (h1 * w1)
            actual_ratio = t2 / t1
            efficiency = expected_ratio / actual_ratio * 100

            print(f"  {h1}x{w1} -> {h2}x{w2}: {actual_ratio:.2f}x slower "
                  f"(expected {expected_ratio:.2f}x, efficiency: {efficiency:.1f}%)")

    # Analyze scaling with kernel size (for each image size)
    print("\n" + "-" * 100)
    print("\nKernel Size Scaling (for each image size):")
    print("-" * 100)

    for img_size in IMAGE_SIZES:
        h, w = img_size[0], img_size[1]
        print(f"\n{h}x{w} image:")
        for i in range(len(KERNEL_SIZES) - 1):
            ks1 = KERNEL_SIZES[i]
            ks2 = KERNEL_SIZES[i + 1]
            _, t1 = results[f"{h}x{w}"][ks1]
            _, t2 = results[f"{h}x{w}"][ks2]

            ratio = t2 / t1
            passes_ratio = ks2 / 7 if ks2 > 7 else ks2 / ks1

            print(f"  {ks1}x{ks1} -> {ks2}x{ks2}: {ratio:.2f}x slower "
                  f"(passes ratio: ~{passes_ratio:.1f}x)")

    # Compute throughput summary
    print("\n" + "=" * 100)
    print("THROUGHPUT SUMMARY (Megapixels/second)")
    print("=" * 100)
    print(f"{'Image Size':<15}", end="")
    for ks in KERNEL_SIZES:
        print(f"  {ks}x{ks:<8}", end="")
    print()
    print("-" * 100)

    for img_size in IMAGE_SIZES:
        h, w = img_size[0], img_size[1]
        pixels = h * w
        print(f"{h}x{w:<10}", end="")

        for ks in KERNEL_SIZES:
            _, sinter_time = results[f"{h}x{w}"][ks]
            mpixels = pixels / sinter_time / 1000
            print(f"  {mpixels:>7.1f} ", end="")
        print()

    print()


if __name__ == "__main__":
    run_benchmarks()
