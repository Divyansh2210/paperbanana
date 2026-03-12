# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with model APIs, image processing, and PDF handling.
All generative model calls are routed through OpenRouter.
"""

import json
import re
import asyncio
import base64
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import aiofiles
import httpx
from PIL import Image
from google.genai import types
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default

# OpenRouter client — used for all generative model calls (text + image)
openrouter_api_key = get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", "")
if openrouter_api_key:
    openrouter_client = AsyncOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://paperbanana.app",
            "X-Title": "PaperBanana",
        },
    )
    print("Initialized OpenRouter Client")
else:
    print("Warning: Could not initialize OpenRouter Client. Missing openrouter_api_key.")
    openrouter_client = None

anthropic_api_key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
if anthropic_api_key:
    anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
    print("Initialized Anthropic Client with API Key")
else:
    print("Warning: Could not initialize Anthropic Client. Missing credentials.")
    anthropic_client = None

openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
if openai_api_key:
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    print("Initialized OpenAI Client with API Key")
else:
    print("Warning: Could not initialize OpenAI Client. Missing credentials.")
    openai_client = None


def _convert_to_openrouter_messages(
    contents: List[Dict[str, Any]], system_instruction: str = ""
) -> List[Dict[str, Any]]:
    """Build an OpenAI-compatible messages list from the generic content list."""
    user_content = _convert_to_openai_format(contents)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": user_content})
    return messages


def _extract_image_b64_from_openrouter_response(data: dict) -> str | None:
    """
    Extract a base64-encoded image string from an OpenRouter JSON response.
    Handles both the standard content-list format and the custom 'images' field
    that some OpenRouter image models return.
    """
    try:
        message = data["choices"][0]["message"]

        # Format 1: content is a list of typed blocks
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    m = re.search(r"base64,(.+)", url, re.DOTALL)
                    if m:
                        return m.group(1).strip()

        # Format 2: custom 'images' list (OpenRouter-specific)
        images = message.get("images")
        if images:
            url = images[0].get("image_url", {}).get("url", "")
            m = re.search(r"base64,(.+)", url, re.DOTALL)
            if m:
                return m.group(1).strip()

        # Format 3: content is a plain data-URL string
        if isinstance(content, str) and content.startswith("data:image"):
            m = re.search(r"base64,(.+)", content, re.DOTALL)
            if m:
                return m.group(1).strip()
    except (KeyError, IndexError, TypeError):
        pass
    return None


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    Route all generative model calls through OpenRouter.
    Accepts the same interface as before (google.genai types.GenerateContentConfig)
    so all agents work without changes.
    Handles both text and image generation based on config.response_modalities.
    """
    if openrouter_client is None:
        raise RuntimeError(
            "OpenRouter client was not initialized. "
            "Please set openrouter_api_key in configs/model_config.yaml."
        )

    is_image_gen = bool(
        getattr(config, "response_modalities", None)
        and "IMAGE" in config.response_modalities
    )
    system_instruction = getattr(config, "system_instruction", None) or ""
    temperature = getattr(config, "temperature", 1.0)
    max_tokens = getattr(config, "max_output_tokens", 8192)
    target_candidate_count = getattr(config, "candidate_count", 1)

    messages = _convert_to_openrouter_messages(contents, system_instruction)
    result_list = []

    for attempt in range(max_attempts):
        try:
            if is_image_gen:
                # Image generation — use httpx directly to access raw JSON
                # (OpenRouter image responses may use custom fields not in the OpenAI schema)
                image_config = getattr(config, "image_config", None)
                extra_body: Dict[str, Any] = {"modalities": ["image", "text"]}
                if image_config:
                    if getattr(image_config, "aspect_ratio", None):
                        extra_body["aspect_ratio"] = image_config.aspect_ratio
                    if getattr(image_config, "image_size", None):
                        extra_body["image_size"] = image_config.image_size

                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **extra_body,
                }
                async with httpx.AsyncClient(timeout=120) as http:
                    resp = await http.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openrouter_api_key}",
                            "HTTP-Referer": "https://paperbanana.app",
                            "X-Title": "PaperBanana",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )

                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

                image_b64 = _extract_image_b64_from_openrouter_response(resp.json())
                if image_b64:
                    result_list.append(image_b64)
                    break
                else:
                    print(f"[Warning]: No image in response, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue

            else:
                # Text generation — use AsyncOpenAI client
                response = await openrouter_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    result_list.append(text)
                    if len(result_list) >= target_candidate_count:
                        break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 30)
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")

    if not result_list:
        result_list = ["Error"] * target_candidate_count
    elif len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list[:target_candidate_count]

def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.
    
    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    
    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                # OpenAI expects data URL format
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list

async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    This follows the same pattern as Claude's implementation.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            # If we reach here, the input is valid.
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")
    
    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    
    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            
            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]
