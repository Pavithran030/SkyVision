"""Image validation module for sky photographs."""

from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from PIL import Image


@dataclass
class ImageValidationResult:
    """Structured result of image validation."""
    is_valid: bool
    width: int = 0
    height: int = 0
    channels: int = 0
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    image: Optional[Image.Image] = None


def validate_image_file(
    image_input: Union[Path, str, bytes, io.BytesIO, Image.Image],
    min_width: int = 64,
    min_height: int = 64,
    min_mean_intensity: float = 15.0,
    max_mean_intensity: float = 245.0,
    min_std_intensity: float = 5.0,
) -> ImageValidationResult:
    """
    Validate an image for sky pattern recognition.
    
    Checks:
    - Decodability & file corruption
    - Minimum image dimensions
    - Color channels (converts to RGB)
    - Mean pixel intensity (flags pitch black or overexposed images)
    - Pixel standard deviation (flags blank/solid frames)
    """
    img: Optional[Image.Image] = None
    try:
        if isinstance(image_input, Image.Image):
            img = image_input.copy()
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            img = Image.open(image_input)
        else:
            path = Path(image_input)
            if not path.exists():
                return ImageValidationResult(
                    is_valid=False,
                    error_message=f"Image file does not exist: {path}",
                )
            if path.stat().st_size == 0:
                return ImageValidationResult(
                    is_valid=False,
                    error_message=f"Image file is empty (0 bytes): {path}",
                )
            img = Image.open(path)

        # Force load image to ensure file isn't truncated/corrupt
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")

    except Exception as e:
        return ImageValidationResult(
            is_valid=False,
            error_message=f"Corrupt or unreadable image: {str(e)}",
        )

    width, height = img.size
    if width < min_width or height < min_height:
        return ImageValidationResult(
            is_valid=False,
            width=width,
            height=height,
            error_message=f"Image dimensions ({width}x{height}) are smaller than required minimum ({min_width}x{min_height}).",
        )

    # Compute intensity statistics
    arr = np.asarray(img, dtype=np.float32)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    warnings: List[str] = []
    if mean_val < min_mean_intensity:
        warnings.append(
            f"Image is unusually dark (mean intensity {mean_val:.1f}/255). It may be a night capture or underexposed."
        )
    elif mean_val > max_mean_intensity:
        warnings.append(
            f"Image is unusually bright (mean intensity {mean_val:.1f}/255). It may be washed out or pointing directly at the sun."
        )

    if std_val < min_std_intensity:
        warnings.append(
            f"Image has low contrast / uniform color (intensity std {std_val:.1f}). Features may be indistinct."
        )

    return ImageValidationResult(
        is_valid=True,
        width=width,
        height=height,
        channels=3,
        mean_intensity=mean_val,
        std_intensity=std_val,
        warnings=warnings,
        error_message=None,
        image=img,
    )
