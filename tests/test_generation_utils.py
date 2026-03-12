"""
Tests for utils/generation_utils.py

Covers:
- call_gemini_with_retry_async routes through OpenRouter for text generation
- call_gemini_with_retry_async routes through OpenRouter for image generation
- Image base64 extraction from all OpenRouter response formats
- Retry logic and error handling
- RuntimeError when OpenRouter client is missing
"""

import sys
import json
import base64
import pytest
import httpx
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


PATCH_OPENROUTER_CLIENT = "utils.generation_utils.openrouter_client"
PATCH_OPENROUTER_KEY    = "utils.generation_utils.openrouter_api_key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_config(system="", temperature=1.0, candidate_count=1, max_tokens=100):
    from google.genai import types
    return types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        candidate_count=candidate_count,
        max_output_tokens=max_tokens,
    )


def _make_image_config(aspect_ratio="16:9", image_size="2k"):
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=1.0,
        candidate_count=1,
        max_output_tokens=8192,
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        ),
    )


def _fake_b64_image() -> str:
    from io import BytesIO
    from PIL import Image
    img = Image.new("RGB", (8, 8), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _openrouter_text_response(text: str):
    """Minimal mock for openai ChatCompletion text response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _httpx_image_response(b64: str, fmt: str = "content_list") -> MagicMock:
    """Build a mock httpx.Response with image payload in the given format."""
    data_url = f"data:image/png;base64,{b64}"

    if fmt == "content_list":
        message = {"content": [{"type": "image_url", "image_url": {"url": data_url}}]}
    elif fmt == "images_field":
        message = {"content": None, "images": [{"image_url": {"url": data_url}}]}
    elif fmt == "plain_string":
        message = {"content": data_url}
    else:
        message = {"content": None}

    payload = {"choices": [{"message": message}]}
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = payload
    return mock_resp


# ---------------------------------------------------------------------------
# _extract_image_b64_from_openrouter_response
# ---------------------------------------------------------------------------

def test_extract_image_content_list_format():
    from utils.generation_utils import _extract_image_b64_from_openrouter_response
    b64 = _fake_b64_image()
    data = {"choices": [{"message": {
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
    }}]}
    assert _extract_image_b64_from_openrouter_response(data) == b64


def test_extract_image_images_field_format():
    from utils.generation_utils import _extract_image_b64_from_openrouter_response
    b64 = _fake_b64_image()
    data = {"choices": [{"message": {
        "content": None,
        "images": [{"image_url": {"url": f"data:image/png;base64,{b64}"}}]
    }}]}
    assert _extract_image_b64_from_openrouter_response(data) == b64


def test_extract_image_plain_string_format():
    from utils.generation_utils import _extract_image_b64_from_openrouter_response
    b64 = _fake_b64_image()
    data = {"choices": [{"message": {"content": f"data:image/png;base64,{b64}"}}]}
    assert _extract_image_b64_from_openrouter_response(data) == b64


def test_extract_image_returns_none_when_no_image():
    from utils.generation_utils import _extract_image_b64_from_openrouter_response
    data = {"choices": [{"message": {"content": "just some text, no image"}}]}
    assert _extract_image_b64_from_openrouter_response(data) is None


def test_extract_image_returns_none_on_malformed_response():
    from utils.generation_utils import _extract_image_b64_from_openrouter_response
    assert _extract_image_b64_from_openrouter_response({}) is None
    assert _extract_image_b64_from_openrouter_response({"choices": []}) is None


# ---------------------------------------------------------------------------
# call_gemini_with_retry_async — text generation via OpenRouter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_call_uses_openrouter_client():
    """Text generation must use the openrouter_client, not the Gemini SDK."""
    import utils.generation_utils as gu

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_openrouter_text_response("generated description")
    )

    with patch.object(gu, "openrouter_client", mock_client):
        result = await gu.call_gemini_with_retry_async(
            model_name="google/gemini-3-pro-preview",
            contents=[{"type": "text", "text": "describe this"}],
            config=_make_text_config(),
        )

    assert result == ["generated description"]
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_text_call_passes_model_name_with_prefix():
    """OpenRouter expects the full 'google/...' provider prefix — must NOT be stripped."""
    import utils.generation_utils as gu

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_openrouter_text_response("ok")
    )

    with patch.object(gu, "openrouter_client", mock_client):
        await gu.call_gemini_with_retry_async(
            model_name="google/gemini-3-pro-preview",
            contents=[{"type": "text", "text": "hi"}],
            config=_make_text_config(),
        )

    called_model = mock_client.chat.completions.create.call_args.kwargs["model"]
    assert called_model == "google/gemini-3-pro-preview"


@pytest.mark.asyncio
async def test_text_call_passes_system_instruction():
    """system_instruction from config must be sent as the system message."""
    import utils.generation_utils as gu

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_openrouter_text_response("ok")
    )

    with patch.object(gu, "openrouter_client", mock_client):
        await gu.call_gemini_with_retry_async(
            model_name="google/gemini-3-pro-preview",
            contents=[{"type": "text", "text": "hi"}],
            config=_make_text_config(system="You are a diagram expert."),
        )

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are a diagram expert."}


# ---------------------------------------------------------------------------
# call_gemini_with_retry_async — image generation via OpenRouter (httpx)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_image_call_uses_httpx_not_openai_client():
    """Image generation must use httpx (raw HTTP), not the AsyncOpenAI client."""
    import utils.generation_utils as gu

    mock_openrouter = MagicMock()
    mock_openrouter.chat.completions.create = AsyncMock()  # must NOT be called

    b64 = _fake_b64_image()
    mock_resp = _httpx_image_response(b64, fmt="content_list")
    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_resp)

    with patch.object(gu, "openrouter_client", mock_openrouter), \
         patch.object(gu, "openrouter_api_key", "test-key"), \
         patch("httpx.AsyncClient", return_value=mock_http_client):
        result = await gu.call_gemini_with_retry_async(
            model_name="google/gemini-3-pro-image-preview",
            contents=[{"type": "text", "text": "generate a diagram"}],
            config=_make_image_config(),
        )

    mock_openrouter.chat.completions.create.assert_not_called()
    assert result == [b64]


@pytest.mark.asyncio
async def test_image_call_passes_aspect_ratio_and_size():
    """aspect_ratio and image_size from image_config must reach the httpx payload."""
    import utils.generation_utils as gu

    b64 = _fake_b64_image()
    mock_resp = _httpx_image_response(b64)
    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_resp)

    with patch.object(gu, "openrouter_client", MagicMock()), \
         patch.object(gu, "openrouter_api_key", "test-key"), \
         patch("httpx.AsyncClient", return_value=mock_http_client):
        await gu.call_gemini_with_retry_async(
            model_name="google/gemini-3-pro-image-preview",
            contents=[{"type": "text", "text": "gen"}],
            config=_make_image_config(aspect_ratio="21:9", image_size="4k"),
        )

    payload = mock_http_client.post.call_args.kwargs["json"]
    assert payload["aspect_ratio"] == "21:9"
    assert payload["image_size"] == "4k"
    assert payload["modalities"] == ["image", "text"]


@pytest.mark.asyncio
async def test_image_call_handles_images_field_format():
    """Image extraction must succeed when response uses the 'images' field format."""
    import utils.generation_utils as gu

    b64 = _fake_b64_image()
    mock_resp = _httpx_image_response(b64, fmt="images_field")
    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_resp)

    with patch.object(gu, "openrouter_client", MagicMock()), \
         patch.object(gu, "openrouter_api_key", "test-key"), \
         patch("httpx.AsyncClient", return_value=mock_http_client):
        result = await gu.call_gemini_with_retry_async(
            model_name="google/gemini-3-pro-image-preview",
            contents=[{"type": "text", "text": "gen"}],
            config=_make_image_config(),
        )

    assert result == [b64]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_raises_when_no_openrouter_client():
    """RuntimeError must be raised when openrouter_client is None."""
    import utils.generation_utils as gu

    with patch.object(gu, "openrouter_client", None):
        with pytest.raises(RuntimeError, match="OpenRouter client was not initialized"):
            await gu.call_gemini_with_retry_async(
                model_name="google/gemini-3-pro-preview",
                contents=[{"type": "text", "text": "hi"}],
                config=_make_text_config(),
            )


@pytest.mark.asyncio
async def test_retries_on_text_failure_then_succeeds():
    """Transient failures are retried; succeeds on second attempt."""
    import utils.generation_utils as gu

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[Exception("transient"), _openrouter_text_response("success")]
    )

    with patch.object(gu, "openrouter_client", mock_client), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        result = await gu.call_gemini_with_retry_async(
            model_name="google/gemini-3-pro-preview",
            contents=[{"type": "text", "text": "hi"}],
            config=_make_text_config(),
            max_attempts=3,
            retry_delay=1,
        )

    assert result == ["success"]
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_returns_error_after_all_attempts_fail():
    """Returns ['Error'] after all retry attempts are exhausted."""
    import utils.generation_utils as gu

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("persistent failure")
    )

    with patch.object(gu, "openrouter_client", mock_client), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        result = await gu.call_gemini_with_retry_async(
            model_name="google/gemini-3-pro-preview",
            contents=[{"type": "text", "text": "hi"}],
            config=_make_text_config(),
            max_attempts=2,
            retry_delay=1,
        )

    assert result == ["Error"]
