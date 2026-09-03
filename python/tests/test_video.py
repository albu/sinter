import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sinter import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    RandomCrop,
    Crop,
    Brightness,
    Contrast,
    Gamma,
    Rotate,
)


def test_video_clip_numpy_hwc():
    """Test 4D NumPy array [T, H, W, C]."""
    clip = np.random.randint(0, 256, (8, 128, 128, 3), dtype=np.uint8)
    pipe = Compose([
        HorizontalFlip(p=1.0),
        Brightness(delta=20),
        Crop(x=10, y=10, width=64, height=64),
    ])

    out = pipe.apply_video(clip, seed=42)
    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 64, 64, 3)
    assert out.dtype == np.uint8


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_video_clip_torch_chw():
    """Test 4D PyTorch tensor in standard [T, C, H, W] layout."""
    clip_np = np.random.randint(0, 256, (6, 100, 100, 3), dtype=np.uint8)
    clip_torch = torch.from_numpy(clip_np).permute(0, 3, 1, 2).contiguous()

    pipe = Compose([
        HorizontalFlip(p=1.0),
        Brightness(delta=15),
        Crop(x=5, y=5, width=50, height=50),
    ])

    out_torch = pipe.apply_video(clip_torch, seed=123)
    out_np = pipe.apply_video(clip_np, seed=123)

    assert isinstance(out_torch, torch.Tensor)
    assert out_torch.shape == (6, 3, 50, 50)
    assert out_torch.dtype == torch.uint8

    # Verify bit-exact agreement between PyTorch and NumPy layouts
    out_torch_hwc = out_torch.permute(0, 2, 3, 1).numpy()
    assert np.array_equal(out_np, out_torch_hwc)


def test_video_clip_temporal_consistency():
    """Verify that all frames in a video clip receive the exact same spatial transform."""
    t, h, w, c = 10, 200, 200, 3
    # Create identical frames with a distinct asymmetric marker
    base_frame = np.zeros((h, w, c), dtype=np.uint8)
    base_frame[20:60, 30:80, :] = 200
    clip = np.stack([base_frame.copy() for _ in range(t)], axis=0)

    pipe = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomCrop(width=100, height=100),
    ])

    out = pipe.apply_video(clip, seed=999)
    assert out.shape == (t, 100, 100, c)

    # Every single frame in the clip must have been cropped and flipped identically
    first_frame = out[0]
    for frame_idx in range(1, t):
        assert np.array_equal(first_frame, out[frame_idx]), (
            f"Frame {frame_idx} diverged from Frame 0: temporal spatial consistency broken!"
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_video_batch_5d_torch():
    """Test 5D PyTorch batch [B, T, C, H, W]."""
    b, t, c, h, w = 2, 4, 3, 80, 80
    batch_torch = torch.randint(0, 256, (b, t, c, h, w), dtype=torch.uint8)

    pipe = Compose([
        HorizontalFlip(p=0.5),
        Brightness(delta=10),
        Crop(x=0, y=0, width=60, height=60),
    ])

    out = pipe.apply_video_batch(batch_torch, seed=42)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (b, t, c, 60, 60)
    assert out.dtype == torch.uint8


def test_video_list_of_frames():
    """Test passing a Python list of frames."""
    frames = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
    pipe = Compose([
        Brightness(delta=20),
        Crop(x=4, y=4, width=32, height=32),
    ])

    out = pipe.apply_video(frames, seed=1)
    assert isinstance(out, list)
    assert len(out) == 5
    for f in out:
        assert isinstance(f, np.ndarray)
        assert f.shape == (32, 32, 3)


def test_single_transform_apply_video():
    """Test calling .apply_video() directly on a transform node."""
    clip = np.random.randint(0, 256, (4, 50, 50, 3), dtype=np.uint8)
    flip = HorizontalFlip(p=1.0)
    out = flip.apply_video(clip)
    assert out.shape == (4, 50, 50, 3)

    # Frame 0 reversed along horizontal axis matches output
    assert np.array_equal(out[0], clip[0, :, ::-1, :])


def test_sampled_program_apply_video():
    """Test calling .apply_video() on an already sampled program."""
    clip = np.random.randint(0, 256, (4, 64, 64, 3), dtype=np.uint8)
    pipe = Compose([
        Brightness(delta=25),
        Crop(x=0, y=0, width=32, height=32),
    ])

    sampled = pipe.sample()
    out = sampled.apply_video(clip)
    assert out.shape == (4, 32, 32, 3)


def test_video_reproducibility_with_seed():
    """Test that specifying the same seed yields bit-exact identical video output."""
    clip = np.random.randint(0, 256, (8, 120, 120, 3), dtype=np.uint8)
    pipe = Compose([
        HorizontalFlip(p=0.5),
        RandomCrop(width=80, height=80),
        Brightness(delta=(-30, 30)),
    ])

    out1 = pipe.apply_video(clip, seed=777)
    out2 = pipe.apply_video(clip, seed=777)
    assert np.array_equal(out1, out2)
