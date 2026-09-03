import time
import cv2
import numpy as np
import sinter as sin

def measure(fn, runs=25, warmup=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1000

def main():
    path = "assets/test_baseline.jpg"
    w, h, c = sin.read_header(path)
    print(f"=========================================================================================")
    print(f"  SINTER NATIVE JPEG ENGINE BENCHMARK: {w}x{h} ({w*h/1e6:.1f} Megapixels, RGB)")
    print(f"=========================================================================================\n")

    # 1. Header parsing
    t_header = measure(lambda: sin.read_header(path), runs=100)
    print(f"1. Header Inspection (width, height, channels):")
    print(f"   - Sinter read_header:  {t_header * 1000:.1f} µs ({1000 / t_header:,.0f} headers/sec)")
    print()

    # 2. Full Image Decode to RGB
    t_cv_full = measure(lambda: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    t_sin_full = measure(lambda: sin.imread(path))
    print(f"2. Full Image Decode to RGB8 (2592x1632):")
    print(f"   - OpenCV (cv2.imread + cvtColor BGR2RGB): {t_cv_full:.2f} ms")
    print(f"   - Sinter (pure Rust native decode to RGB): {t_sin_full:.2f} ms")
    print()

    # 3. ROI Crop Decoding (Random Crop 224x224 - Standard ViT / ResNet input)
    t_cv_crop224 = measure(lambda: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)[500:724, 500:724])
    t_sin_crop224 = measure(lambda: sin.imread(path, crop=(500, 500, 224, 224)))
    speedup_224 = t_cv_crop224 / t_sin_crop224
    print(f"3. Region-of-Interest Crop: 224x224 (Standard ViT / ResNet DL Input):")
    print(f"   - OpenCV (Decode full 4.2MP + BGR2RGB + slice): {t_cv_crop224:.2f} ms")
    print(f"   - Sinter (Native MCU block-skipping ROI crop):   {t_sin_crop224:.2f} ms")
    print(f"   --> Speedup vs OpenCV: {speedup_224:.2f}x faster ({1000/t_sin_crop224:,.0f} crops/sec)")
    print()

    # 4. ROI Crop Decoding (Random Crop 256x256)
    t_cv_crop256 = measure(lambda: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)[500:756, 500:756])
    t_sin_crop256 = measure(lambda: sin.imread(path, crop=(500, 500, 256, 256)))
    speedup_256 = t_cv_crop256 / t_sin_crop256
    print(f"4. Region-of-Interest Crop: 256x256 Crop from 4.2MP Image:")
    print(f"   - OpenCV (Decode full 4.2MP + BGR2RGB + slice): {t_cv_crop256:.2f} ms")
    print(f"   - Sinter (Native MCU block-skipping ROI crop):   {t_sin_crop256:.2f} ms")
    print(f"   --> Speedup vs OpenCV: {speedup_256:.2f}x faster ({1000/t_sin_crop256:,.0f} crops/sec)")
    print()

    # 4. ROI Crop Decoding (512x512)
    t_cv_crop512 = measure(lambda: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)[300:812, 300:812])
    t_sin_crop512 = measure(lambda: sin.imread(path, crop=(300, 300, 512, 512)))
    speedup_512 = t_cv_crop512 / t_sin_crop512
    print(f"4. Region-of-Interest Crop: 512x512 Crop from 4.2MP Image:")
    print(f"   - OpenCV (Decode full 4.2MP + BGR2RGB + slice): {t_cv_crop512:.2f} ms")
    print(f"   - Sinter (Native MCU block-skipping ROI crop):   {t_sin_crop512:.2f} ms")
    print(f"   --> Speedup vs OpenCV: {speedup_512:.2f}x faster ({1000/t_sin_crop512:,.0f} crops/sec)")
    print()
    print(f"=========================================================================================")

if __name__ == "__main__":
    main()
