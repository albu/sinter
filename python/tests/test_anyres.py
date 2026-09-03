import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sinter import AnyRes


def test_anyres_grid_selection():
    anyres = AnyRes(tile_size=448, max_tiles=6, include_thumbnail=True)

    # 16:9 widescreen (1920x1080)
    assert anyres.select_grid(1920, 1080) == (2, 1)

    # 9:16 portrait (1080x1920)
    assert anyres.select_grid(1080, 1920) == (1, 2)

    # 1:1 square (1000x1000)
    assert anyres.select_grid(1000, 1000) == (2, 2)

    # 4:1 panorama (2000x500)
    assert anyres.select_grid(2000, 500) == (4, 1)


def test_anyres_numpy_hwc():
    anyres = AnyRes(tile_size=128, max_tiles=4, include_thumbnail=True)
    img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

    # 1:1 square -> (2, 2) grid = 4 tiles + 1 thumbnail = 5 tiles
    tiles = anyres(img)
    assert isinstance(tiles, np.ndarray)
    assert tiles.shape == (5, 128, 128, 3)
    assert tiles.dtype == np.uint8


def test_anyres_without_thumbnail():
    anyres = AnyRes(tile_size=64, max_tiles=4, include_thumbnail=False)
    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    # (2, 2) grid without thumbnail = 4 tiles
    tiles = anyres(img)
    assert tiles.shape == (4, 64, 64, 3)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_anyres_torch_chw():
    anyres = AnyRes(tile_size=128, max_tiles=4, include_thumbnail=True)
    img_np = np.random.randint(0, 256, (200, 400, 3), dtype=np.uint8)
    img_torch = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

    tiles_np = anyres(img_np)
    tiles_torch = anyres(img_torch)

    assert isinstance(tiles_torch, torch.Tensor)
    assert tiles_torch.dtype == torch.uint8
    # Shape is [N, C, S, S]
    assert tiles_torch.shape == (tiles_np.shape[0], 3, 128, 128)

    # Verify bit-exact match between PyTorch CHW and NumPy HWC
    tiles_torch_hwc = tiles_torch.permute(0, 2, 3, 1).numpy()
    assert np.array_equal(tiles_np, tiles_torch_hwc)


def test_anyres_interpolation_modes():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    for interp in ["nearest", "bilinear", "bicubic", "lanczos4"]:
        anyres = AnyRes(tile_size=64, max_tiles=2, include_thumbnail=True, interpolation=interp)
        tiles = anyres(img)
        assert tiles.ndim == 4
        assert tiles.shape[1:3] == (64, 64)
