"""Sinter: High-Performance Compiled Image Augmentation Engine.

Pure Rust + SIMD image transformation engine featuring automatic kernel and LUT fusion,
zero-copy multi-target processing (images, bboxes, keypoints, masks), and first-class
serializable sampled IR programs.
"""

from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
import numpy as np

# Type aliases
DistInput = Union[
    float,
    int,
    Tuple[float, float],
    List[float],
    "Constant",
    "Uniform",
    "UniformInt",
    "Bernoulli",
    "Normal",
]

BBoxFormat = Literal[
    "xyxy",
    "xywh",
    "cxcywh",
    "rel_xyxy",
    "rel_xywh",
    "rel_cxcywh",
    "pascal_voc",
    "coco",
    "albumentations",
    "yolo",
]

KeypointFormat = Literal[
    "xy",
    "xyv",
    "rel_xy",
    "rel_xyv",
]

InterpolationMode = Union[
    Literal["nearest", "bilinear", "bicubic", "lanczos4"],
    "Interpolation",
]

PadModeType = Union[
    Literal["reflect", "replicate", "wrap", "constant"],
    "PadMode",
]

RotateAngleType = Union[
    Literal[90, 180, 270, "90", "180", "270"],
    "RotateAngle",
]

EmbossDirectionType = Union[
    Literal[
        "top_left",
        "top",
        "top_right",
        "right",
        "bottom_right",
        "bottom",
        "bottom_left",
        "left",
    ],
    "EmbossDirection",
]

EdgeMethodType = Union[
    Literal["sobel", "prewitt", "laplacian", "canny"],
    "EdgeMethod",
]

# =============================================================================
# Distributions
# =============================================================================

class Constant:
    """Fixed constant parameter value (deterministic)."""
    value: float
    def __init__(self, value: float) -> None: ...
    def __repr__(self) -> str: ...

class Uniform:
    """Uniform continuous random distribution in [min, max]."""
    min: float
    max: float
    def __init__(self, min: float, max: float) -> None: ...
    def __repr__(self) -> str: ...

class UniformInt:
    """Uniform discrete integer random distribution in [min, max]."""
    min: int
    max: int
    def __init__(self, min: int, max: int) -> None: ...
    def __repr__(self) -> str: ...

class Bernoulli:
    """Bernoulli random variable with probability p of success."""
    p: float
    def __init__(self, p: float) -> None: ...
    def __repr__(self) -> str: ...

class Normal:
    """Gaussian normal distribution with mean mu and standard deviation sigma."""
    mu: float
    sigma: float
    def __init__(self, mu: float, sigma: float) -> None: ...
    def __repr__(self) -> str: ...

# =============================================================================
# Enums
# =============================================================================

class RotateAngle:
    ROTATE_90: RotateAngle
    ROTATE_180: RotateAngle
    ROTATE_270: RotateAngle

class Interpolation:
    NEAREST: Interpolation
    BILINEAR: Interpolation
    BICUBIC: Interpolation
    LANCZOS4: Interpolation

class PadMode:
    REFLECT: PadMode
    REPLICATE: PadMode
    WRAP: PadMode
    @staticmethod
    def constant(value: int = 0) -> PadMode: ...

class EmbossDirection:
    TOP_LEFT: EmbossDirection
    TOP: EmbossDirection
    TOP_RIGHT: EmbossDirection
    RIGHT: EmbossDirection
    BOTTOM_RIGHT: EmbossDirection
    BOTTOM: EmbossDirection
    BOTTOM_LEFT: EmbossDirection
    LEFT: EmbossDirection

class EdgeMethod:
    SOBEL: EdgeMethod
    PREWITT: EdgeMethod
    LAPLACIAN: EdgeMethod
    CANNY: EdgeMethod

# =============================================================================
# Core Pipeline & Sampled Program
# =============================================================================

class SampledImageProgram:
    """Deterministic, optimized compiled execution plan produced by sampling a Compose pipeline.

    All random variables have been resolved into concrete parameters.
    Can be serialized, deserialized, inspected with .explain(), and executed.
    """
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def version(self) -> int: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json_str: str) -> SampledImageProgram: ...
    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(data: bytes) -> SampledImageProgram: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> SampledImageProgram: ...
    def explain(self) -> str: ...
    def summary(self) -> str: ...
    def to_mermaid(self, direction: str = "LR") -> str: ...
    def visualize(self, direction: str = "LR") -> str: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __getitem__(self, index: int) -> str: ...
    def __iter__(self) -> Iterator[str]: ...
    def __call__(
        self,
        image: Union[np.ndarray, Any],
        bboxes: Optional[Union[np.ndarray, Sequence[Any], Any]] = None,
        keypoints: Optional[Union[np.ndarray, Sequence[Any], Any]] = None,
        masks: Optional[Union[np.ndarray, Any]] = None,
        mask: Optional[Union[np.ndarray, Any]] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Dict[str, Any]: ...
    def apply(self, array: Union[np.ndarray, Any], inplace: bool = False) -> Union[np.ndarray, Any]: ...
    def apply_batch(
        self,
        images: Union[Sequence[Any], np.ndarray, Any],
        inplace: bool = False,
        num_threads: Optional[int] = None,
    ) -> Union[List[Any], np.ndarray, Any]: ...
    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        image_size: Tuple[int, int],
        format: BBoxFormat = "xywh",
        format_out: Optional[BBoxFormat] = None,
    ) -> np.ndarray: ...
    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        image_size: Tuple[int, int],
        format: KeypointFormat = "xy",
        format_out: Optional[KeypointFormat] = None,
    ) -> np.ndarray: ...
    def apply_to_masks(
        self,
        mask: Union[np.ndarray, Any],
        image_size: Tuple[int, int],
        inplace: bool = False,
    ) -> Union[np.ndarray, Any]: ...
    def apply_to_labels(
        self,
        labels: np.ndarray,
        image_size: Tuple[int, int],
    ) -> np.ndarray: ...

# Internal alias
_SampledImageProgram = SampledImageProgram

class Compose:
    """Pipeline of image augmentations.

    Supports automatic operator fusion, zero-copy multi-target transformations
    (images, bboxes, keypoints, masks), and deterministic sampling.
    """
    def __init__(self, transforms: Optional[Sequence[Any]] = None) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __getitem__(self, index: Union[int, slice]) -> Union[Any, "Compose"]: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __add__(self, other: Union["Compose", Sequence[Any], Any]) -> "Compose": ...
    def __radd__(self, other: Union[Sequence[Any], Any]) -> "Compose": ...
    def explain(self, seed: Optional[int] = None) -> str: ...
    def summary(self, seed: Optional[int] = None) -> str: ...
    def to_json(self, seed: Optional[int] = None) -> str: ...
    def to_mermaid(self, seed: Optional[int] = None, direction: str = "LR") -> str: ...
    def visualize(self, seed: Optional[int] = None, direction: str = "LR") -> str: ...
    def sample(self, seed: Optional[int] = None) -> SampledImageProgram: ...
    def sample_with_seed(self, seed: int) -> SampledImageProgram: ...
    def __call__(
        self,
        image: Union[np.ndarray, Any],
        bboxes: Optional[Union[np.ndarray, Sequence[Any], Any]] = None,
        keypoints: Optional[Union[np.ndarray, Sequence[Any], Any]] = None,
        masks: Optional[Union[np.ndarray, Any]] = None,
        mask: Optional[Union[np.ndarray, Any]] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        seed: Optional[int] = None,
        inplace: bool = False,
    ) -> Dict[str, Any]: ...
    def apply(
        self,
        array: Union[np.ndarray, Any],
        inplace: bool = False,
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, Any]: ...
    def apply_batch(
        self,
        images: Union[Sequence[Any], np.ndarray, Any],
        inplace: bool = False,
        num_threads: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Union[List[Any], np.ndarray, Any]: ...

# =============================================================================
# Geometric Transforms
# =============================================================================

# Call convention: transform(image) returns the transformed ndarray;
# transform(image, bboxes=..., masks=...) returns a dict of targets,
# as does Compose.__call__.
class HorizontalFlip:
    """Flip the image horizontally around the y-axis."""
    p: DistInput
    def __init__(self, p: Optional[DistInput] = None) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class VerticalFlip:
    """Flip the image vertically around the x-axis."""
    p: DistInput
    def __init__(self, p: Optional[DistInput] = None) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Transpose:
    """Transpose image dimensions (swap height and width)."""
    p: DistInput
    def __init__(self, p: Optional[DistInput] = None) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Rotate:
    """Rotate image by 90, 180, or 270 degrees."""
    angle: RotateAngleType
    p: DistInput
    def __init__(
        self,
        angle: Optional[RotateAngleType] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Resize:
    """Resize image to exact (width, height) dimensions."""
    width: int
    height: int
    interpolation: InterpolationMode
    p: DistInput
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        interpolation: Optional[InterpolationMode] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Crop:
    """Crop image region [x, y, x + width, y + height]."""
    x: DistInput
    y: DistInput
    width: DistInput
    height: DistInput
    p: DistInput
    def __init__(
        self,
        x: Optional[DistInput] = None,
        y: Optional[DistInput] = None,
        width: Optional[DistInput] = None,
        height: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Pad:
    """Pad image borders with specified mode (reflect, replicate, wrap, constant)."""
    top: DistInput
    bottom: DistInput
    left: DistInput
    right: DistInput
    mode: PadModeType
    value: int
    p: DistInput
    def __init__(
        self,
        top: Optional[DistInput] = None,
        bottom: Optional[DistInput] = None,
        left: Optional[DistInput] = None,
        right: Optional[DistInput] = None,
        mode: Optional[PadModeType] = None,
        value: Optional[int] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Affine:
    """Affine transform (scale, rotate, translate, shear)."""
    scale: Tuple[DistInput, DistInput]
    rotate: DistInput
    translate: Tuple[DistInput, DistInput]
    shear: Tuple[DistInput, DistInput]
    interpolation: InterpolationMode
    border_mode: Union[PadMode, str, int]
    p: DistInput
    def __init__(
        self,
        scale: Optional[DistInput] = None,
        rotate: Optional[DistInput] = None,
        translate: Optional[DistInput] = None,
        shear: Optional[DistInput] = None,
        interpolation: Optional[InterpolationMode] = None,
        border_mode: Optional[Union[PadMode, str, int]] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

# =============================================================================
# Photometric Transforms
# =============================================================================

class Brightness:
    """Adjust image brightness by adding delta."""
    delta: DistInput
    p: DistInput
    def __init__(
        self,
        delta: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Contrast:
    """Adjust image contrast by multiplying factor."""
    factor: DistInput
    p: DistInput
    def __init__(
        self,
        factor: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Invert:
    """Invert pixel values (255 - pixel)."""
    p: DistInput
    def __init__(self, p: Optional[DistInput] = None) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Posterize:
    """Reduce number of bits for each color channel."""
    bits: DistInput
    p: DistInput
    def __init__(
        self,
        bits: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Solarize:
    """Invert all pixel values above a threshold."""
    threshold: DistInput
    p: DistInput
    def __init__(
        self,
        threshold: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Gamma:
    """Apply gamma correction."""
    gamma: DistInput
    p: DistInput
    def __init__(
        self,
        gamma: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Normalize:
    """Normalize pixel values with mean and standard deviation.

    Fast uint8-preserving scale transform: `(pixel - mean) / std * 255`.
    For float32 standardized output tensors, convert after augmentation.
    """
    mean: DistInput
    std: DistInput
    p: DistInput
    def __init__(
        self,
        mean: Optional[DistInput] = None,
        std: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    @staticmethod
    def standard() -> Normalize: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Equalize:
    """Histogram equalization."""
    p: DistInput
    def __init__(self, p: Optional[DistInput] = None) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class AutoContrast:
    """Maximize image contrast by stretching histogram."""
    cutoff: float
    p: DistInput
    def __init__(self, cutoff: float = 0.0, p: Optional[DistInput] = None) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class ToGray:
    """Convert RGB image to grayscale."""
    p: DistInput
    def __init__(self, p: Optional[DistInput] = None) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class ToSepia:
    """Apply sepia color filter."""
    p: DistInput
    def __init__(self, p: Optional[DistInput] = None) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class ToRGB:
    """Convert grayscale image to RGB."""
    p: DistInput
    def __init__(self, p: Optional[DistInput] = None) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

# =============================================================================
# Noise & Color Transforms
# =============================================================================

class GaussNoise:
    """Add Gaussian noise to image pixels."""
    mean: DistInput
    std: DistInput
    p: DistInput
    def __init__(
        self,
        mean: Optional[DistInput] = None,
        std: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class MultiplicativeNoise:
    """Multiply image pixels by random factors."""
    multiplier: DistInput
    p: DistInput
    def __init__(
        self,
        multiplier: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class SaltAndPepper:
    """Add salt and pepper noise."""
    amount: DistInput
    salt_vs_pepper: DistInput
    p: DistInput
    def __init__(
        self,
        amount: Optional[DistInput] = None,
        salt_vs_pepper: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class RGBShift:
    """Independently shift R, G, B channels."""
    r_shift: DistInput
    g_shift: DistInput
    b_shift: DistInput
    p: DistInput
    def __init__(
        self,
        r_shift: Optional[DistInput] = None,
        g_shift: Optional[DistInput] = None,
        b_shift: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class HueSaturationValue:
    """Randomly change hue, saturation and value."""
    hue_shift: DistInput
    saturation_scale: DistInput
    value_scale: DistInput
    p: DistInput
    def __init__(
        self,
        hue_shift: Optional[DistInput] = None,
        saturation_scale: Optional[DistInput] = None,
        value_scale: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class ColorTemperature:
    """Adjust color temperature (warm / cool)."""
    temperature: DistInput
    p: DistInput
    def __init__(
        self,
        temperature: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class ColorBalance:
    """Scale R, G, B channels independently."""
    r_scale: DistInput
    g_scale: DistInput
    b_scale: DistInput
    p: DistInput
    def __init__(
        self,
        r_scale: Optional[DistInput] = None,
        g_scale: Optional[DistInput] = None,
        b_scale: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class ColorTint:
    """Blend image with a specific tint color and alpha."""
    tint: Sequence[DistInput]
    p: DistInput
    def __init__(
        self,
        tint: Optional[Sequence[DistInput]] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class ChannelShuffle:
    """Permute color channels."""
    order: Union[int, Literal["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"]]
    p: DistInput
    def __init__(
        self,
        order: Optional[Union[int, str]] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    @staticmethod
    def rgb() -> ChannelShuffle: ...
    @staticmethod
    def rbg() -> ChannelShuffle: ...
    @staticmethod
    def grb() -> ChannelShuffle: ...
    @staticmethod
    def gbr() -> ChannelShuffle: ...
    @staticmethod
    def brg() -> ChannelShuffle: ...
    @staticmethod
    def bgr() -> ChannelShuffle: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

# =============================================================================
# Dropout Transforms
# =============================================================================

class CoarseDropout:
    """Dropout rectangular regions."""
    holes: DistInput
    hole_size: Union[DistInput, Tuple[DistInput, DistInput]]
    p: DistInput
    def __init__(
        self,
        holes: Optional[DistInput] = None,
        hole_size: Optional[Union[DistInput, Tuple[DistInput, DistInput]]] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class GridDropout:
    """Grid-based dropout."""
    ratio: DistInput
    unit_size: DistInput
    holes: DistInput
    p: DistInput
    def __init__(
        self,
        ratio: Optional[DistInput] = None,
        unit_size: Optional[DistInput] = None,
        holes: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

# =============================================================================
# Kernel / Convolution Transforms
# =============================================================================

class GaussianBlur:
    """Gaussian blur with discrete kernel sizes (3, 5, 7, 13, 21, 31)."""
    kernel_size: int
    p: DistInput
    def __init__(
        self,
        kernel_size: Optional[int] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class GaussianBlurSigma:
    """Continuous sigma Gaussian blur with exact/fast kernel calculation."""
    sigma: float
    quality: Literal["Exact", "Fast"]
    p: DistInput
    def __init__(
        self,
        sigma: float = 1.0,
        quality: Optional[str] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class MedianBlur:
    """Median blur filter."""
    kernel_size: int
    p: DistInput
    def __init__(
        self,
        kernel_size: Optional[int] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Sharpen:
    """Sharpen image."""
    strength: DistInput
    p: DistInput
    def __init__(
        self,
        strength: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class Emboss:
    """Emboss image with directional highlights."""
    direction: EmbossDirectionType
    alpha: DistInput
    strength: DistInput
    p: DistInput
    def __init__(
        self,
        direction: Optional[EmbossDirectionType] = None,
        alpha: Optional[DistInput] = None,
        strength: Optional[DistInput] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

class EdgeDetection:
    """Edge detection filters (Sobel, Prewitt, Laplacian, Canny)."""
    method: EdgeMethodType
    p: DistInput
    def __init__(
        self,
        method: Optional[EdgeMethodType] = None,
        p: Optional[DistInput] = None,
    ) -> None: ...
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        bbox_format: BBoxFormat = "xywh",
        keypoint_format: KeypointFormat = "xy",
        inplace: bool = False,
    ) -> Union[np.ndarray, Dict[str, Any]]: ...
    def apply(self, array: np.ndarray, inplace: bool = False) -> np.ndarray: ...

# =============================================================================
# Batch Transforms
# =============================================================================

class MixUp:
    alpha: float
    def __init__(self, alpha: float = 0.2) -> None: ...

class CutMix:
    alpha: float
    def __init__(self, alpha: float = 1.0) -> None: ...

class Mosaic:
    output_size: Tuple[int, int]
    def __init__(self, output_size: Tuple[int, int]) -> None: ...

class BatchPipeline:
    def __init__(self, transforms: Sequence[Any]) -> None: ...

# =============================================================================
# PyTorch utilities
# =============================================================================

def apply_to_tensor_inplace(tensor: Any, compose: Compose) -> None:
    """Apply a Compose pipeline to a PyTorch tensor in-place (zero-copy for CPU tensors).

    Requires uint8 HWC CPU tensor. Raises ValueError if pipeline changes tensor shape.
    """
    ...
