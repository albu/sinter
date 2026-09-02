import numpy as np
import pytest
import sinter
from sinter import (
    Choice,
    Compose,
    GaussianBlur,
    HorizontalFlip,
    Identity,
    MedianBlur,
    OneOf,
    VerticalFlip,
)
import torch


def test_identity_transform():
    img = np.arange(64, dtype=np.uint8).reshape(8, 8)
    ident = Identity()
    out = ident(img)
    np.testing.assert_array_equal(out, img)
    assert ident.apply(img) is not None


def test_choice_basic():
    img = np.ones((16, 16, 3), dtype=np.uint8) * 100
    choice = Choice([GaussianBlur(3), MedianBlur(3)], p=1.0)
    assert len(choice) == 2
    res = choice(img)
    assert res.shape == (16, 16, 3)


def test_choice_weights():
    # Only HorizontalFlip with weight 1.0 vs VerticalFlip with weight 0.0
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    choice = Choice([HorizontalFlip(p=1.0), VerticalFlip(p=1.0)], weights=[1.0, 0.0])
    res = choice(img)
    np.testing.assert_array_equal(res, np.fliplr(img))

    # Reverse weights
    choice2 = Choice([HorizontalFlip(p=1.0), VerticalFlip(p=1.0)], weights=[0.0, 1.0])
    res2 = choice2(img)
    np.testing.assert_array_equal(res2, np.flipud(img))


def test_choice_validation():
    with pytest.raises(ValueError, match="at least one"):
        Choice([])

    with pytest.raises(ValueError, match="weights length"):
        Choice([GaussianBlur(3)], weights=[0.5, 0.5])

    with pytest.raises(ValueError, match="non-negative"):
        Choice([GaussianBlur(3)], weights=[-1.0])

    with pytest.raises(ValueError, match="non-negative"):
        Choice([GaussianBlur(3), MedianBlur(3)], weights=[0.0, 0.0])


def test_oneof_is_choice():
    assert OneOf is Choice


def test_choice_in_compose():
    p = Compose([
        Choice([HorizontalFlip(p=1.0), Identity()], weights=[1.0, 0.0]),
    ])
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    res = p(image=img)
    np.testing.assert_array_equal(res["image"], np.fliplr(img))


def test_compose_configured_bbox_and_keypoint_format():
    # Pipeline-level formats
    p = Compose(
        [HorizontalFlip(p=1.0)],
        bbox_format="pascal_voc",
        keypoint_format="xy",
    )
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 30, 40]], dtype=np.float32)  # x1, y1, x2, y2
    kpts = np.array([[20, 25]], dtype=np.float32)

    # Call WITHOUT passing bbox_format or keypoint_format
    res = p(image=img, bboxes=boxes, keypoints=kpts)
    assert "bboxes" in res
    assert "keypoints" in res

    # Check horizontal flip on pascal_voc [x1, y1, x2, y2]:
    # x1_new = 100 - x2 = 70, x2_new = 100 - x1 = 90
    expected_box = np.array([[70, 10, 90, 40]], dtype=np.float32)
    np.testing.assert_allclose(res["bboxes"], expected_box)


def test_singular_bbox_and_keypoint_aliases():
    p = Compose([HorizontalFlip(p=1.0)], bbox_format="pascal_voc")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    box = np.array([[10, 10, 30, 40]], dtype=np.float32)
    kpt = np.array([[20, 25]], dtype=np.float32)

    res = p(image=img, bbox=box, keypoint=kpt)
    assert "bbox" in res
    assert "bboxes" not in res
    assert "keypoint" in res
    assert "keypoints" not in res


def test_kwargs_passthrough():
    p = Compose([HorizontalFlip(p=1.0)])
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = p(
        image=img,
        sample_id=42,
        filepath="dataset/train/001.jpg",
        metadata={"category": "cat"},
        labels=[1, 2, 3],  # image-level labels pass through when no bboxes
    )
    assert res["sample_id"] == 42
    assert res["filepath"] == "dataset/train/001.jpg"
    assert res["metadata"] == {"category": "cat"}
    assert res["labels"] == [1, 2, 3]


def test_bbox_labels_guard():
    p = Compose([HorizontalFlip(p=1.0)])
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 20, 20]], dtype=np.float32)
    with pytest.raises(TypeError, match="labels is not a separate argument"):
        p(image=img, bboxes=boxes, labels=[0])


def test_4d_batch_auto_dispatch():
    p = Compose([HorizontalFlip(p=1.0)])
    # 4D numpy array (B, H, W, C)
    batch_np = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    batch_np[:, :, :16, :] = 255
    res_call = p(image=batch_np)
    assert "image" in res_call
    assert res_call["image"].shape == (4, 32, 32, 3)
    np.testing.assert_array_equal(res_call["image"][0, :, 16:, :], 255)

    # 4D numpy array via apply()
    res_apply = p.apply(batch_np)
    assert res_apply.shape == (4, 32, 32, 3)
    np.testing.assert_array_equal(res_apply[0, :, 16:, :], 255)

    # 4D PyTorch tensor (B, C, H, W)
    batch_torch = torch.zeros((4, 3, 32, 32), dtype=torch.uint8)
    batch_torch[:, :, :, :16] = 255
    res_torch = p.apply(batch_torch)
    assert isinstance(res_torch, torch.Tensor)
    assert res_torch.shape == (4, 3, 32, 32)
    assert torch.all(res_torch[0, :, :, 16:] == 255)
