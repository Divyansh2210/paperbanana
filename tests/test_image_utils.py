"""
Tests for utils/image_utils.py

Covers:
- convert_png_b64_to_jpg_b64 happy path: RGB and RGBA PNG inputs
- Output is a valid JPEG base64 string
- Edge cases: None, empty string, short string, invalid base64, non-image bytes
"""

import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import convert_png_b64_to_jpg_b64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_b64(mode: str = "RGB", size: tuple = (16, 16), color=(100, 150, 200)) -> str:
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _is_valid_jpeg_b64(b64_str: str) -> bool:
    try:
        data = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(data))
        return img.format == "JPEG"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_converts_rgb_png_to_jpeg():
    result = convert_png_b64_to_jpg_b64(_make_png_b64(mode="RGB"))
    assert result is not None
    assert _is_valid_jpeg_b64(result)


def test_converts_rgba_png_to_jpeg():
    """RGBA PNG must be converted to RGB before JPEG encoding; no OSError."""
    result = convert_png_b64_to_jpg_b64(_make_png_b64(mode="RGBA", color=(100, 150, 200, 128)))
    assert result is not None
    assert _is_valid_jpeg_b64(result)


def test_output_differs_from_input():
    """JPEG base64 output must differ from PNG base64 input (different format)."""
    png_b64 = _make_png_b64()
    result = convert_png_b64_to_jpg_b64(png_b64)
    assert result != png_b64


def test_output_is_non_empty():
    """JPEG output must have non-trivial size."""
    result = convert_png_b64_to_jpg_b64(_make_png_b64(size=(100, 100)))
    assert result is not None
    assert len(base64.b64decode(result)) > 100


def test_large_image_converts_successfully():
    result = convert_png_b64_to_jpg_b64(_make_png_b64(size=(512, 512)))
    assert result is not None
    assert _is_valid_jpeg_b64(result)


def test_single_pixel_image_converts():
    result = convert_png_b64_to_jpg_b64(_make_png_b64(size=(1, 1)))
    assert result is not None
    assert _is_valid_jpeg_b64(result)


# ---------------------------------------------------------------------------
# Edge cases — invalid / degenerate inputs
# ---------------------------------------------------------------------------

def test_returns_none_for_none():
    assert convert_png_b64_to_jpg_b64(None) is None


def test_returns_none_for_empty_string():
    assert convert_png_b64_to_jpg_b64("") is None


def test_returns_none_for_string_shorter_than_10_chars():
    assert convert_png_b64_to_jpg_b64("abc") is None
    assert convert_png_b64_to_jpg_b64("123456789") is None  # exactly 9 chars


def test_accepts_string_of_exactly_10_chars_but_still_fails_gracefully():
    """10-char non-image base64 must return None, not raise."""
    result = convert_png_b64_to_jpg_b64("a" * 10)
    # Either returns None (not valid image) or raises internally and returns None
    assert result is None


def test_returns_none_for_invalid_base64():
    assert convert_png_b64_to_jpg_b64("NOT-VALID-BASE64-!!@@##$%^&*()") is None


def test_returns_none_for_valid_base64_non_image():
    non_image_b64 = base64.b64encode(b"this is plain text, definitely not a PNG").decode()
    assert convert_png_b64_to_jpg_b64(non_image_b64) is None


def test_returns_none_for_truncated_png():
    """Truncated PNG header is valid base64 but invalid image data."""
    png_b64 = _make_png_b64()
    truncated = png_b64[:len(png_b64) // 2]
    assert convert_png_b64_to_jpg_b64(truncated) is None
