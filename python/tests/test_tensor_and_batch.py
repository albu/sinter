import numpy as np
import pytest

# torch is a test-only dependency (tensor interop tests); skip gracefully
# when absent so CI wheel builds need only numpy + pytest.
torch = pytest.importorskip("torch")

pytest.importorskip("sinter")

from sinter import (
    Compose,
    Brightness,
    Contrast,
    Gamma,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    Resize,
    Uniform,
)


class TestPyTorchTensorIngestion:
    """Test native PyTorch tensor handling (HWC, CHW, 2D) without user conversion ceremony."""

    def test_tensor_hwc_uint8(self):
        pipeline = Compose([Brightness(delta=50)])
        tensor = torch.full((64, 64, 3), 100, dtype=torch.uint8)

        # Apply pipeline directly to tensor
        res = pipeline.apply(tensor)

        assert isinstance(res, torch.Tensor)
        assert res.shape == (64, 64, 3)
        assert res.dtype == torch.uint8
        assert res[0, 0, 0].item() == 150
        # Original was not mutated by default
        assert tensor[0, 0, 0].item() == 100

    def test_tensor_chw_uint8(self):
        pipeline = Compose([Brightness(delta=50), HorizontalFlip()])
        # Standard PyTorch CHW tensor (Channels=3, Height=64, Width=64)
        tensor = torch.zeros((3, 64, 64), dtype=torch.uint8)
        tensor[:, :, :10] = 200  # Left strip

        res = pipeline.apply(tensor)

        assert isinstance(res, torch.Tensor)
        assert res.shape == (3, 64, 64)  # Preserves CHW layout!
        assert res.dtype == torch.uint8
        # Left strip moved to right due to HorizontalFlip, and increased brightness by 50 (250)
        assert res[0, 0, -5].item() == 250
        assert res[0, 0, 5].item() == 50

    def test_tensor_call_with_targets(self):
        pipeline = Compose([HorizontalFlip()])
        tensor = torch.zeros((3, 100, 100), dtype=torch.uint8)
        mask_tensor = torch.zeros((100, 100), dtype=torch.uint8)
        mask_tensor[:, :20] = 1

        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)

        res = pipeline(tensor, bboxes=bboxes, masks=mask_tensor)

        assert isinstance(res["image"], torch.Tensor)
        assert res["image"].shape == (3, 100, 100)
        assert isinstance(res["masks"], torch.Tensor)
        assert res["masks"].shape == (100, 100)
        assert torch.all(res["masks"][:, -20:] == 1)


class TestMermaidVisualization:
    """Test Mermaid flowchart generation from execution plans."""

    def test_to_mermaid_output(self):
        pipeline = Compose([
            Brightness(delta=20),
            Contrast(factor=1.2),
            Gamma(gamma=0.9),
            HorizontalFlip(),
            VerticalFlip(),
        ])

        mermaid_str = pipeline.to_mermaid()

        assert "flowchart LR" in mermaid_str
        assert "Input([" in mermaid_str
        assert "Output([" in mermaid_str
        assert "Pass" in mermaid_str

        # Test sampled program mermaid
        sampled = pipeline.sample_with_seed(42)
        sample_mermaid = sampled.to_mermaid(direction="TD")
        assert "flowchart TD" in sample_mermaid


class TestRayonBatchParallelism:
    """Test multi-core batch processing with Rayon."""

    def test_apply_batch_4d_numpy(self):
        pipeline = Compose([
            Brightness(delta=50),
            HorizontalFlip(),
        ])

        batch_size = 16
        images = np.ones((batch_size, 32, 32, 3), dtype=np.uint8) * 100
        original = images.copy()

        # Run multi-threaded batch apply
        results = pipeline.apply_batch(images, num_threads=4)

        assert isinstance(results, np.ndarray)
        assert results.shape == (batch_size, 32, 32, 3)
        assert results[0, 0, 0, 0] == 150
        # Check immutability
        np.testing.assert_array_equal(images, original)

    def test_apply_batch_4d_tensor_nchw(self):
        pipeline = Compose([
            Brightness(delta=30),
        ])

        batch_size = 8
        # (N, C, H, W)
        tensors = torch.full((batch_size, 3, 32, 32), 100, dtype=torch.uint8)

        results = pipeline.apply_batch(tensors, num_threads=4)

        assert isinstance(results, torch.Tensor)
        assert results.shape == (batch_size, 3, 32, 32)
        assert results[0, 0, 0, 0].item() == 130

    def test_apply_batch_list_of_arrays(self):
        pipeline = Compose([
            Brightness(delta=Uniform(-50, 50)),
        ])

        img_list = [np.full((32, 32, 3), 100, dtype=np.uint8) for _ in range(10)]
        results = pipeline.apply_batch(img_list, num_threads=4, seed=123)

        assert isinstance(results, list)
        assert len(results) == 10
        # Since Uniform distribution is used, distinct items in batch get different seeds
        means = [res.mean() for res in results]
        assert len(set(means)) > 1  # Verify statistical variation across batch items!


class TestConditionalInputCopy:
    """Test that out-of-place pipelines (Resize, Pad, Affine) skip redundant input copies."""

    def test_resize_preserves_input_without_unnecessary_copy(self):
        pipeline = Compose([Resize(32, 32)])
        img = np.ones((64, 64, 3), dtype=np.uint8) * 100
        original = img.copy()

        # default inplace=False: does not mutate input and produces 32x32 output
        res = pipeline.apply(img)
        assert res.shape == (32, 32, 3)
        np.testing.assert_array_equal(img, original)

    def test_pipeline_starting_with_barrier_then_inplace(self):
        # Resize creates a BarrierImage, and subsequent Brightness mutates the BarrierImage.
        # Original input 'img' is never touched.
        pipeline = Compose([Resize(32, 32), Brightness(delta=50)])
        img = np.ones((64, 64, 3), dtype=np.uint8) * 100
        original = img.copy()

        res = pipeline.apply(img)
        assert res.shape == (32, 32, 3)
        assert res[0, 0, 0] == 150
        np.testing.assert_array_equal(img, original)


class TestAffineBorderMode:
    """Test Affine transform with configurable border_mode."""

    def test_affine_border_mode_modes(self):
        from sinter import Affine, PadMode

        # Test string literals
        a1 = Affine(translate=(10, 10), border_mode="reflect")
        assert "border_mode='reflect'" in repr(a1)

        a2 = Affine(translate=(10, 10), border_mode="replicate")
        assert "border_mode='replicate'" in repr(a2)

        a3 = Affine(translate=(10, 10), border_mode="wrap")
        assert "border_mode='wrap'" in repr(a3)

        a4 = Affine(translate=(10, 10), border_mode="constant")
        assert "border_mode='constant(0)'" in repr(a4)

        # Test PadMode enum
        a5 = Affine(translate=(10, 10), border_mode=PadMode.REFLECT)
        assert "border_mode='reflect'" in repr(a5)

