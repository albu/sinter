"""Tests for Normalize: real float32 semantics (no u8 clamping)."""

import numpy as np
import pytest

import sinter


@pytest.fixture
def img():
    rng = np.random.default_rng(11)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


def test_output_is_float32_and_matches_reference(img):
    mean, std = 0.45, 0.22
    out = sinter.Normalize(mean=mean, std=std)(img)
    assert out.dtype == np.float32
    expected = (img.astype(np.float32) / 255.0 - mean) / std
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)


def test_no_precision_loss_at_extremes():
    # The old u8-LUT Normalize clamped 45% of pixels to 0. The float32
    # version must keep every value distinct and unclamped.
    img = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    out = sinter.Normalize(mean=0.45, std=0.22)(img)
    assert np.unique(out).size == 256
    assert out.min() < -2.0 and out.max() > 2.0


def test_standard_normalization():
    img = np.full((8, 8, 3), 128, dtype=np.uint8)
    out = sinter.Normalize.standard()(img)
    np.testing.assert_allclose(out, 128.0 / 255.0, atol=1e-7)


def test_must_be_last_transform(img):
    p = sinter.Compose([sinter.Normalize(mean=0.45, std=0.22), sinter.Brightness(delta=10)])
    with pytest.raises(ValueError) as ei:
        p.apply(img)
    assert "last" in str(ei.value)


def test_mid_pipeline_position_ok_when_last(img):
    # Photometric ops BEFORE Normalize are fine (and fuse into one pass)
    p = sinter.Compose([sinter.Brightness(delta=20), sinter.Normalize(mean=0.5, std=0.25)])
    out = p.apply(img)
    assert out.dtype == np.float32
    expected = (np.clip(img.astype(np.float32) + 20, 0, 255) / 255.0 - 0.5) / 0.25
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-5)


def test_input_not_modified(img):
    original = img.copy()
    sinter.Normalize(mean=0.5, std=0.25).apply(img)
    np.testing.assert_array_equal(img, original)


def test_with_bboxes_dict_form(img):
    bb = np.array([[10.0, 10.0, 20.0, 20.0, 1.0]], dtype=np.float32)
    r = sinter.Normalize(mean=0.5, std=0.5)(img, bboxes=bb)
    assert isinstance(r, dict)
    assert r["image"].dtype == np.float32
    np.testing.assert_array_equal(r["bboxes"], bb)


def test_seeded_reproducibility_with_distributions(img):
    p = sinter.Compose([
        sinter.Normalize(
            mean=sinter.Uniform(0.3, 0.5),
            std=sinter.Uniform(0.2, 0.3),
        )
    ])
    a = p(img, seed=7)["image"]
    b = p(img, seed=7)["image"]
    np.testing.assert_array_equal(a, b)
