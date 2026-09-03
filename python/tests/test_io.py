import pytest
import numpy as np
import sinter as sin

def test_read_header():
    w, h, c = sin.read_header("assets/test_baseline.jpg")
    assert w == 2592
    assert h == 1632
    assert c == 3

    # Also works on progressive header without decoding pixels
    w2, h2, c2 = sin.read_header("assets/logo.jpg")
    assert w2 == 2592
    assert h2 == 1632
    assert c2 == 3

def test_imread_full_and_crop():
    path = "assets/test_baseline.jpg"
    full = sin.imread(path)
    assert full.shape == (1632, 2592, 3)
    assert full.dtype == np.uint8

    # ROI Crop at (200, 300, 128, 128)
    crop = sin.imread(path, crop=(200, 300, 128, 128))
    assert crop.shape == (128, 128, 3)
    assert crop.dtype == np.uint8

    # Check bit-exact match with full image slice
    np.testing.assert_array_equal(crop, full[300:300+128, 200:200+128])

def test_imread_corner_crops():
    path = "assets/test_baseline.jpg"
    full = sin.imread(path)

    # Top-left corner
    tl = sin.imread(path, crop=(0, 0, 64, 64))
    np.testing.assert_array_equal(tl, full[0:64, 0:64])

    # Crop with non-multiple of 16 dimensions (e.g. 77x53)
    odd = sin.imread(path, crop=(150, 210, 77, 53))
    assert odd.shape == (53, 77, 3)
    np.testing.assert_array_equal(odd, full[210:210+53, 150:150+77])

def test_decode_jpeg_bytes():
    with open("assets/test_baseline.jpg", "rb") as f:
        data = f.read()

    full_from_bytes = sin.decode_jpeg(data)
    assert full_from_bytes.shape == (1632, 2592, 3)

    crop_from_bytes = sin.decode_jpeg(data, crop=(100, 100, 64, 64))
    np.testing.assert_array_equal(crop_from_bytes, full_from_bytes[100:164, 100:164])

def test_crop_out_of_bounds():
    path = "assets/test_baseline.jpg"
    with pytest.raises(Exception):
        sin.imread(path, crop=(2500, 1600, 200, 200))
