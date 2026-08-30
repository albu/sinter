"""
Gaussian Blur Correctness Tests

Tests both theoretical properties (mathematical invariants) and
empirical validation (comparison with OpenCV).
"""

import numpy as np
import sys

try:
    import cv2
    cv2.setNumThreads(0)
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not installed, skipping comparison tests")

from sinter import GaussianBlur, Compose


def test_constant_image():
    """Constant image should remain constant (kernel sums to 1)."""
    print("\n1. Testing constant image preservation...")
    for size in [10, 50, 100]:
        for kernel_size in [3, 7, 13, 31]:
            data = np.full((size, size, 3), 128, dtype=np.uint8)
            pipe = Compose([GaussianBlur(kernel_size=kernel_size)])
            result = pipe.apply(data.copy())

            assert np.allclose(result, 128, atol=1), f"{size}x{size}, kernel {kernel_size}x{kernel_size}"

    print("   ✅ PASS: Constant images remain constant")


def test_impulse_response():
    """Impulse (single bright pixel) should produce the kernel shape."""
    print("\n2. Testing impulse response (kernel shape)...")

    # Create a black image with a single white pixel in center
    size = 21
    data = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    data[center, center] = 255

    # Apply 7x7 Gaussian
    pipe = Compose([GaussianBlur(kernel_size=7)])
    result = pipe.apply(data.copy())

    # The result should be symmetric around center
    center_cross = result[center, :, 0]  # Horizontal cross-section

    # Check symmetry: left and right should be mirror images
    left = center_cross[:center]
    right = center_cross[center+1:][::-1]  # Reverse for comparison

    assert np.allclose(left, right, atol=2), f"Kernel not symmetric: Left={left}, Right={right}"
    assert center_cross[center] >= center_cross[center-1] and center_cross[center] >= center_cross[center+1], f"Center not brightest: {center_cross}"

    for i in range(center - 1):
        assert center_cross[i] <= center_cross[i+1], f"Not monotonically decreasing: {center_cross}"

    print(f"   ✅ PASS: Impulse response is symmetric and peaked at center")


def test_mean_preservation():
    """Gaussian blur should preserve mean brightness."""
    print("\n3. Testing mean preservation...")

    np.random.seed(42)
    for kernel_size in [3, 7, 13, 31]:
        data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        original_mean = data.mean()

        pipe = Compose([GaussianBlur(kernel_size=kernel_size)])
        result = pipe.apply(data.copy())

        new_mean = result.mean()
        assert abs(new_mean - original_mean) <= 1.0, f"Kernel {kernel_size}x{kernel_size} mean diff > 1.0"

    print("   ✅ PASS: Mean brightness preserved")


def test_linearity():
    """GaussianBlur(a*x + b*y) should equal a*GaussianBlur(x) + b*GaussianBlur(y)."""
    print("\n4. Testing linearity...")

    np.random.seed(42)
    x = np.random.randint(0, 128, (50, 50, 3), dtype=np.uint8)
    y = np.random.randint(0, 128, (50, 50, 3), dtype=np.uint8)

    pipe = Compose([GaussianBlur(kernel_size=7)])

    # Compute blur of sum
    sum_xy = (x.astype(np.float32) + y.astype(np.float32)) / 2.0
    sum_xy = sum_xy.astype(np.uint8)
    blur_of_sum = pipe.apply(sum_xy.copy()).astype(np.float32)

    # Compute sum of blurs
    blur_x = pipe.apply(x.copy()).astype(np.float32)
    blur_y = pipe.apply(y.copy()).astype(np.float32)
    sum_of_blurs = (blur_x + blur_y) / 2.0

    assert np.allclose(blur_of_sum, sum_of_blurs, atol=2.0), "Linearity violated"
    print("   ✅ PASS: Blur is approximately linear")


def test_variance_attenuation():
    """Variance should strictly decrease after blurring (smoothing property)."""
    print("\n5. Testing variance attenuation (smoothing)...")

    np.random.seed(42)
    data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    original_var = data.var()

    prev_var = original_var
    for kernel_size in [3, 5, 7]:
        pipe = Compose([GaussianBlur(kernel_size=kernel_size)])
        result = pipe.apply(data.copy())
        new_var = result.var()

        assert new_var < prev_var, f"Variance did not decrease for {kernel_size}: {prev_var} -> {new_var}"
        prev_var = new_var

    print("   ✅ PASS: Variance strictly decreases with kernel size")


def test_opencv_comparison():
    """Compare Sinter Gaussian Blur with OpenCV."""
    if not HAS_CV2:
        print("\n6. Skipping OpenCV comparison (cv2 not available)")
        return True

    print("\n6. Comparing with OpenCV implementation...")

    test_images = {
        "Random Noise": np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        "Gradient": np.linspace(0, 255, 10000, dtype=np.uint8).reshape(100, 100)[:, :, np.newaxis].repeat(3, axis=2),
        "Checkerboard": (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2),
    }

    for name, img in test_images.items():
        # Sinter
        pipe = Compose([GaussianBlur(kernel_size=7)])
        our_result = pipe.apply(img.copy())

        # OpenCV
        cv_result = cv2.GaussianBlur(img, (7, 7), 0)

        # Compare
        mse = np.mean((our_result.astype(np.float32) - cv_result.astype(np.float32)) ** 2)
        assert mse < 100, f"{name}: MSE={mse:.2f} > 100"

    print(f"   ✅ PASS: Comparison complete (within expected tolerances)")


def test_pascal_row_6():
    """Verify 7x7 kernel uses correct Pascal weights."""
    print("\n7. Testing 7x7 Pascal row 6 weights...")

    # Create impulse response
    size = 15
    data = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    data[center, center] = 255

    pipe = Compose([GaussianBlur(kernel_size=7)])
    result = pipe.apply(data.copy())

    cross_section = result[center, :, 0].astype(np.float32)
    cross_section = cross_section / cross_section.sum()  # Normalize

    expected = np.array([1, 6, 15, 20, 15, 6, 1], dtype=np.float32) / 64.0
    center_idx = size // 2
    actual = cross_section[center_idx - 3:center_idx + 4]

    assert np.allclose(actual, expected, atol=0.08), f"Weights differ: Expected {expected}, Actual {actual}"
    print(f"   ✅ PASS: Weights match Pascal row 6")


def run_all_tests():
    """Run all correctness tests."""
    print("=" * 70)
    print("GAUSSIAN BLUR CORRECTNESS TEST SUITE")
    print("=" * 70)

    tests = [
        ("Constant Image", test_constant_image),
        ("Impulse Response", test_impulse_response),
        ("Mean Preservation", test_mean_preservation),
        ("Linearity", test_linearity),
        ("Variance Attenuation", test_variance_attenuation),
        ("OpenCV Comparison", test_opencv_comparison),
        ("Pascal Row 6", test_pascal_row_6),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n   ❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed or had warnings")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
