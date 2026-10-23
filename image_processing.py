"""
Image preprocessing module for Handwritten Digit Recognition.
Handles format validation, polarity normalization, bounding box cropping,
aspect-ratio preserving resize to 28x28, and center-of-mass centering.
"""

import io
import os
from typing import Union
import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG"}


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""
    pass


class UnsupportedFormatError(ImageProcessingError):
    """Raised when the uploaded image format is not supported."""
    pass


class InvalidImageError(ImageProcessingError):
    """Raised when the image file is corrupt or cannot be decoded."""
    pass


class BlankImageError(ImageProcessingError):
    """Raised when the image is blank or contains no identifiable digit strokes."""
    pass


def _shift_image(img_arr: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """
    Shift a 28x28 image array by (shift_y, shift_x) pixels with zero padding.
    Clamps shifts to [-4, 4] pixels to prevent shifting the digit off-canvas.
    """
    shift_y = max(-4, min(4, int(round(shift_y))))
    shift_x = max(-4, min(4, int(round(shift_x))))

    if shift_y == 0 and shift_x == 0:
        return img_arr

    shifted = np.zeros_like(img_arr)
    src_y_start = max(0, -shift_y)
    src_y_end = min(28, 28 - shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end = min(28, 28 + shift_y)

    src_x_start = max(0, -shift_x)
    src_x_end = min(28, 28 - shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end = min(28, 28 + shift_x)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img_arr[
        src_y_start:src_y_end, src_x_start:src_x_end
    ]
    return shifted


def _to_png_data_url(pil_img: Image.Image) -> str:
    """Helper to encode a PIL Image as a base64 PNG data URL."""
    import base64
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def preprocess_image(
    image_input: Union[bytes, io.BytesIO, str, os.PathLike, Image.Image],
    return_metadata: bool = False,
) -> Union[np.ndarray, tuple]:
    """
    Preprocess an input image for MNIST CNN inference.

    Steps:
      1. Decode safely from bytes, buffer, file path, or PIL Image.
      2. Enforce supported formats (PNG, JPG, JPEG).
      3. Flatten transparency and convert to grayscale ('L').
      4. Detect blank / uniform images.
      5. Detect background polarity (light-on-dark vs dark-on-light) using border pixels.
         Normalize so foreground digit is bright (>0) and background is black (0).
      6. Detect digit bounding box and reject images with negligible foreground.
      7. Resize bounding box to fit inside a 20x20 box preserving aspect ratio.
      8. Center in a 28x28 canvas and adjust by center of mass (MNIST standard).
      9. Normalize pixel values to [0.0, 1.0] and reshape to (1, 28, 28, 1).

    Returns:
      np.ndarray of shape (1, 28, 28, 1) with dtype float32.
    """
    # 1. Open and decode image safely
    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input
            img_format = getattr(pil_img, "format", None) or "PNG"
        elif isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise InvalidImageError("Input byte stream is empty.")
            buf = io.BytesIO(image_input)
            pil_img = Image.open(buf)
            img_format = pil_img.format
        elif isinstance(image_input, io.BytesIO):
            pil_img = Image.open(image_input)
            img_format = pil_img.format
        elif isinstance(image_input, (str, os.PathLike)):
            if not os.path.exists(image_input):
                raise InvalidImageError(f"File not found: {image_input}")
            pil_img = Image.open(image_input)
            img_format = pil_img.format
        else:
            raise InvalidImageError(f"Unsupported input type: {type(image_input).__name__}")

        # Force loading image data to catch corrupt image streams
        pil_img.load()
    except (UnidentifiedImageError, OSError, ValueError) as err:
        raise InvalidImageError(f"Failed to decode image: {err}") from err

    # 2. Check supported format
    # Pillow identifies JPEG files with format name 'JPEG'
    format_name = (img_format or "").upper()
    if format_name == "JPG":
        format_name = "JPEG"
    if format_name not in {"PNG", "JPEG"}:
        raise UnsupportedFormatError(
            f"Unsupported image format: '{format_name}'. Supported formats are PNG, JPG, and JPEG."
        )

    # 3. Handle transparency / alpha channels and convert to grayscale
    if pil_img.mode in ("RGBA", "LA") or (
        pil_img.mode == "P" and "transparency" in pil_img.info
    ):
        rgba = pil_img.convert("RGBA")
        # Composite over white background so transparent canvas areas become white
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(bg, rgba)
        gray_img = composite.convert("L")
    else:
        gray_img = pil_img.convert("L")

    raw_arr = np.array(gray_img, dtype=np.float32)

    # 4. Check for blank / uniform image
    if raw_arr.size == 0:
        raise BlankImageError("Image contains no pixels.")
    if np.ptp(raw_arr) < 15.0 or np.std(raw_arr) < 4.0:
        raise BlankImageError("Image is blank or has negligible contrast.")

    # 5. Polarity detection: inspect outer border pixels
    # Avoids blindly inverting every image.
    border_pixels = np.concatenate([
        raw_arr[0, :],      # top row
        raw_arr[-1, :],     # bottom row
        raw_arr[:, 0],      # left column
        raw_arr[:, -1],     # right column
    ])
    border_mean = np.mean(border_pixels)

    if border_mean > 127.0:
        # Dark digit on light background -> invert so background is 0 and digit is >0
        norm_arr = 255.0 - raw_arr
    else:
        # Light digit on dark background -> keep as-is (already matches MNIST)
        norm_arr = raw_arr.copy()

    # 6. Detect digit region / bounding box
    # Filter out low-intensity noise
    threshold = max(30.0, float(np.mean(norm_arr) + 0.3 * np.std(norm_arr)))
    stroke_mask = norm_arr > threshold
    foreground_pixels = np.sum(stroke_mask)

    # Reject if total stroke pixels are negligible
    if foreground_pixels < 20:
        raise BlankImageError("No recognizable digit strokes detected.")

    y_indices, x_indices = np.nonzero(stroke_mask)
    ymin, ymax = int(y_indices.min()), int(y_indices.max()) + 1
    xmin, xmax = int(x_indices.min()), int(x_indices.max()) + 1

    crop_h = ymax - ymin
    crop_w = xmax - xmin

    if crop_h <= 1 or crop_w <= 1:
        raise BlankImageError("Detected digit region is too small.")

    digit_crop = norm_arr[ymin:ymax, xmin:xmax]

    # 7. Resize bounding box into 20x20 box preserving aspect ratio
    scale = 20.0 / max(crop_h, crop_w)
    new_w = max(1, min(20, int(round(crop_w * scale))))
    new_h = max(1, min(20, int(round(crop_h * scale))))

    crop_pil = Image.fromarray(digit_crop.astype(np.uint8))
    resized_pil = crop_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_digit = np.array(resized_pil, dtype=np.float32)

    # 8. Center in 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.float32)
    pad_y = (28 - new_h) // 2
    pad_x = (28 - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_digit

    # Adjust centering by Center of Mass (MNIST convention)
    total_mass = np.sum(canvas)
    shift_y, shift_x = 0.0, 0.0
    if total_mass > 0:
        y_grid, x_grid = np.indices(canvas.shape)
        cy = np.sum(y_grid * canvas) / total_mass
        cx = np.sum(x_grid * canvas) / total_mass
        shift_y = 13.5 - cy
        shift_x = 13.5 - cx
        canvas = _shift_image(canvas, shift_y, shift_x)

    # 9. Normalize to [0.0, 1.0] and reshape to (1, 28, 28, 1)
    # Clip any potential out-of-range artifacts
    canvas = np.clip(canvas, 0.0, 255.0) / 255.0
    processed_tensor = canvas.reshape(1, 28, 28, 1).astype(np.float32)

    if return_metadata:
        # Generate base64 Data URLs for intermediate and final preprocessing previews
        grayscale_preview = _to_png_data_url(gray_img)

        final_uint8 = (canvas * 255.0).astype(np.uint8)
        tensor_preview = _to_png_data_url(Image.fromarray(final_uint8, mode="L"))

        polarity_desc = (
            "Inverted (dark digit on light background)"
            if border_mean > 127.0
            else "Preserved (light digit on dark background)"
        )

        metadata = {
            "grayscale_preview": grayscale_preview,
            "tensor_preview": tensor_preview,
            "polarity": polarity_desc,
            "bounding_box": {"width": crop_w, "height": crop_h},
            "scaled_size": {"width": new_w, "height": new_h},
            "center_of_mass_shift": {"shift_x": round(float(shift_x), 2), "shift_y": round(float(shift_y), 2)},
            "steps": [
                "Converted color/transparency to single-channel grayscale.",
                f"Normalized polarity: {polarity_desc}.",
                f"Cropped digit bounding box ({crop_w}×{crop_h} px) and resized to fit 20×20 box ({new_w}×{new_h} px).",
                f"Centered on 28×28 canvas by Center of Mass (Δx={shift_x:+.1f}, Δy={shift_y:+.1f}).",
                "Normalized pixel intensities to range [0.0, 1.0] for CNN input.",
            ],
        }
        return processed_tensor, metadata

    return processed_tensor
