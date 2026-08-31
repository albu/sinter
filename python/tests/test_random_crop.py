"""Tests for RandomCrop: shape, determinism, label alignment, error paths."""

import numpy as np
import pytest

import sinter


@pytest.fixture
def img():
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (100, 120, 3), dtype=np.uint8)


def test_output_shape(img):
    out = sinter.RandomCrop(64, 48)(img)
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8


def test_full_size_crop_is_identity(img):
    out = sinter.RandomCrop(120, 100)(img)
    np.testing.assert_array_equal(out, img)


def test_seeded_reproducibility(img):
    p = sinter.Compose([sinter.RandomCrop(64, 48)])
    a = p(img, seed=42)["image"]
    b = p(img, seed=42)["image"]
    c = p(img, seed=43)["image"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_positions_vary_across_seeds(img):
    p = sinter.Compose([sinter.RandomCrop(32, 32)])
    firsts = {p(img, seed=s)["image"][0, 0, 0] for s in range(12)}
    assert len(firsts) > 1, "random crop always picked the same position"


def test_bbox_payload_stays_aligned(img):
    bb = np.array(
        [[10.0, 10.0, 20.0, 20.0, 1.0], [60.0, 60.0, 30.0, 30.0, 3.0]],
        dtype=np.float32,
    )
    p = sinter.Compose([sinter.RandomCrop(64, 64)])
    out = p(img, bboxes=bb, seed=1)
    boxes = out["bboxes"]
    assert boxes.shape[1] == 5, "payload columns must survive"
    # Every surviving box lies inside the crop window
    assert np.all(boxes[:, 0] >= 0) and np.all(boxes[:, 2] <= 64)
    assert np.all(boxes[:, 1] >= 0) and np.all(boxes[:, 3] <= 64)


def test_mask_matches_image_geometry(img):
    m = (np.random.default_rng(3).random((100, 120)) > 0.5).astype(np.uint8)
    p = sinter.Compose([sinter.RandomCrop(64, 48)])
    r = p(img, masks=m, seed=5)
    assert r["masks"].shape == (48, 64)


def test_window_larger_than_image_raises(img):
    with pytest.raises(ValueError) as ei:
        sinter.RandomCrop(200, 200).apply(img)
    assert "RandomCrop" in str(ei.value)
    assert "exceeds" in str(ei.value)


def test_zero_window_rejected():
    with pytest.raises(ValueError):
        sinter.RandomCrop(0, 10)


def test_repr():
    assert "RandomCrop(width=64, height=64" in repr(sinter.RandomCrop(64, 64))
