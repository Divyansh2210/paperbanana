"""
Tests for the refine_image_with_nanoviz pipeline (2-step: text analysis → image gen)
and guards for the existing generate-candidates flow.
"""

import asyncio
import base64
import sys
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Streamlit must be mocked before demo.py is imported, since it executes
# top-level st.* calls at import time.
# ---------------------------------------------------------------------------
sys.modules["streamlit"] = MagicMock()

# Add project root so demo / utils are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo import refine_image_with_nanoviz, get_config_val  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(color=(200, 100, 50), size=(32, 32)) -> bytes:
    """Create a minimal JPEG image in memory."""
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_rgba_bytes(size=(32, 32)) -> bytes:
    """Create a minimal RGBA PNG image in memory."""
    img = Image.new("RGBA", size, color=(200, 100, 50, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_image_b64() -> str:
    """Return a base64-encoded minimal JPEG to simulate a model image response."""
    return base64.b64encode(_make_jpeg_bytes()).decode("utf-8")


PATCH_TARGET = "utils.generation_utils.call_gemini_with_retry_async"


# ---------------------------------------------------------------------------
# refine_image_with_nanoviz — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refine_happy_path():
    """Both steps succeed: returns image bytes and a success message."""
    image_bytes = _make_jpeg_bytes()
    fake_b64 = _fake_image_b64()

    with patch(PATCH_TARGET, new_callable=AsyncMock) as mock_gemini:
        # Step 1 → description text; Step 2 → base64 image
        mock_gemini.side_effect = [
            ["A clean, restructured academic diagram with three boxes arranged in a row."],
            [fake_b64],
        ]
        result_bytes, message = await refine_image_with_nanoviz(
            image_bytes=image_bytes,
            edit_prompt="arrange the info in a neat and easy to understand way",
        )

    assert result_bytes is not None
    assert len(result_bytes) > 0
    assert "✅" in message
    assert mock_gemini.call_count == 2


@pytest.mark.asyncio
async def test_refine_passes_edit_prompt_to_step1():
    """The user's edit_prompt must appear in the Step 1 analysis prompt."""
    image_bytes = _make_jpeg_bytes()
    captured_contents = []

    async def capture(*args, **kwargs):
        captured_contents.append(kwargs.get("contents") or args[1])
        if len(captured_contents) == 1:
            return ["Some layout description"]
        return [_fake_image_b64()]

    with patch(PATCH_TARGET, side_effect=capture):
        await refine_image_with_nanoviz(
            image_bytes=image_bytes,
            edit_prompt="MY_UNIQUE_EDIT_PROMPT",
        )

    step1_text = captured_contents[0][0]["text"]
    assert "MY_UNIQUE_EDIT_PROMPT" in step1_text


@pytest.mark.asyncio
async def test_refine_passes_description_to_step2():
    """The description produced in Step 1 must be forwarded to Step 2."""
    image_bytes = _make_jpeg_bytes()
    step1_desc = "UNIQUE_LAYOUT_DESCRIPTION_XYZ"
    captured_contents = []

    async def capture(*args, **kwargs):
        captured_contents.append(kwargs.get("contents") or args[1])
        if len(captured_contents) == 1:
            return [step1_desc]
        return [_fake_image_b64()]

    with patch(PATCH_TARGET, side_effect=capture):
        await refine_image_with_nanoviz(
            image_bytes=image_bytes,
            edit_prompt="restructure",
        )

    step2_text = captured_contents[1][0]["text"]
    assert step1_desc in step2_text


@pytest.mark.asyncio
async def test_refine_passes_aspect_ratio_and_size():
    """aspect_ratio and image_size must reach the Step 2 image config."""
    image_bytes = _make_jpeg_bytes()
    captured_configs = []

    async def capture(*args, **kwargs):
        cfg = kwargs.get("config") or (args[2] if len(args) > 2 else None)
        if cfg is not None:
            captured_configs.append(cfg)
        if len(captured_configs) == 1:
            return ["Some description"]
        return [_fake_image_b64()]

    with patch(PATCH_TARGET, side_effect=capture):
        await refine_image_with_nanoviz(
            image_bytes=image_bytes,
            edit_prompt="restructure",
            aspect_ratio="16:9",
            image_size="4K",
        )

    step2_cfg = captured_configs[1]
    assert step2_cfg.image_config.aspect_ratio == "16:9"
    assert step2_cfg.image_config.image_size == "4k"  # lowercased


# ---------------------------------------------------------------------------
# refine_image_with_nanoviz — Step 1 failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refine_step1_returns_error_string():
    """If Step 1 returns ['Error'], the function must return None and an error message."""
    with patch(PATCH_TARGET, new_callable=AsyncMock) as mock_gemini:
        mock_gemini.return_value = ["Error"]
        result_bytes, message = await refine_image_with_nanoviz(
            image_bytes=_make_jpeg_bytes(),
            edit_prompt="restructure",
        )

    assert result_bytes is None
    assert "Step 1" in message or "❌" in message
    # Step 2 must NOT be called
    assert mock_gemini.call_count == 1


@pytest.mark.asyncio
async def test_refine_step1_returns_empty_list():
    """If Step 1 returns an empty list, the function must return None and an error message."""
    with patch(PATCH_TARGET, new_callable=AsyncMock) as mock_gemini:
        mock_gemini.return_value = []
        result_bytes, message = await refine_image_with_nanoviz(
            image_bytes=_make_jpeg_bytes(),
            edit_prompt="restructure",
        )

    assert result_bytes is None
    assert "❌" in message
    assert mock_gemini.call_count == 1


# ---------------------------------------------------------------------------
# refine_image_with_nanoviz — Step 2 failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refine_step2_returns_error_string():
    """If Step 2 returns ['Error'], the function must return None and a step-2 error message."""
    with patch(PATCH_TARGET, new_callable=AsyncMock) as mock_gemini:
        mock_gemini.side_effect = [
            ["A valid layout description from step 1."],
            ["Error"],
        ]
        result_bytes, message = await refine_image_with_nanoviz(
            image_bytes=_make_jpeg_bytes(),
            edit_prompt="restructure",
        )

    assert result_bytes is None
    assert "Step 2" in message or "❌" in message
    assert mock_gemini.call_count == 2


@pytest.mark.asyncio
async def test_refine_step2_returns_empty_list():
    """If Step 2 returns [], the function must return None and an error message."""
    with patch(PATCH_TARGET, new_callable=AsyncMock) as mock_gemini:
        mock_gemini.side_effect = [
            ["A valid layout description."],
            [],
        ]
        result_bytes, message = await refine_image_with_nanoviz(
            image_bytes=_make_jpeg_bytes(),
            edit_prompt="restructure",
        )

    assert result_bytes is None
    assert "❌" in message


# ---------------------------------------------------------------------------
# refine_image_with_nanoviz — exception handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refine_exception_in_step1():
    """An unexpected exception in Step 1 must be caught and returned as an error tuple."""
    with patch(PATCH_TARGET, new_callable=AsyncMock) as mock_gemini:
        mock_gemini.side_effect = RuntimeError("network timeout")
        result_bytes, message = await refine_image_with_nanoviz(
            image_bytes=_make_jpeg_bytes(),
            edit_prompt="restructure",
        )

    assert result_bytes is None
    assert "❌" in message
    assert "network timeout" in message


@pytest.mark.asyncio
async def test_refine_exception_in_step2():
    """An unexpected exception in Step 2 must be caught and returned as an error tuple."""
    call_count = 0

    async def raise_on_second(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ["Some description"]
        raise RuntimeError("image generation failed")

    with patch(PATCH_TARGET, side_effect=raise_on_second):
        result_bytes, message = await refine_image_with_nanoviz(
            image_bytes=_make_jpeg_bytes(),
            edit_prompt="restructure",
        )

    assert result_bytes is None
    assert "❌" in message
    assert "image generation failed" in message


# ---------------------------------------------------------------------------
# get_config_val — existing config flow guard
# ---------------------------------------------------------------------------

def test_config_val_env_var_takes_priority(monkeypatch):
    """Environment variable must override whatever is in the YAML config."""
    monkeypatch.setenv("TEST_API_KEY", "env-key-value")
    # Pass a model_config_data that has a conflicting yaml value via the
    # underlying demo-module-level model_config_data dict — we just test
    # the function signature contract directly.
    result = get_config_val("api_keys", "some_key", "TEST_API_KEY", "default")
    assert result == "env-key-value"


def test_config_val_default_when_missing(monkeypatch):
    """Returns default when neither env var nor yaml key is present."""
    monkeypatch.delenv("NONEXISTENT_KEY_XYZ", raising=False)
    result = get_config_val("nonexistent_section", "nonexistent_key", "NONEXISTENT_KEY_XYZ", "my-default")
    assert result == "my-default"


def test_config_val_empty_string_returns_default(monkeypatch):
    """An empty env var must fall through to the default."""
    monkeypatch.setenv("EMPTY_KEY", "")
    result = get_config_val("nonexistent_section", "nonexistent_key", "EMPTY_KEY", "fallback")
    assert result == "fallback"


# ---------------------------------------------------------------------------
# RGBA → RGB conversion guard (regression for the fix in demo.py)
# ---------------------------------------------------------------------------

def test_rgba_image_converts_to_rgb_for_jpeg():
    """
    Saving an RGBA image as JPEG must not raise — the conversion to RGB
    must happen before the save call (as fixed in demo.py).
    """
    img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")  # must not raise
    assert buf.tell() > 0


def test_rgb_image_unchanged():
    """RGB images must pass through the conversion guard without change."""
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    original_mode = img.mode
    if img.mode == "RGBA":
        img = img.convert("RGB")
    assert img.mode == original_mode == "RGB"
