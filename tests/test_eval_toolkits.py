"""
Tests for utils/eval_toolkits.py

Covers:
- _determine_tier_outcome: all agreement/conflict/neutral combinations
- _try_regex_extract_winner: all 5 regex patterns + edge cases
- get_score_for_image_referenced: missing GT path, missing eval_image_field,
  Tier 1 decides, Tier 1 tie → Tier 2 decides, all-neutral → Tie
"""

import base64
import io
import sys
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.eval_toolkits import (
    _determine_tier_outcome,
    _try_regex_extract_winner,
    get_score_for_image_referenced,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_b64(size=(8, 8), color=(100, 100, 100)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _write_gt_image(tmp_path, task_name="diagram", filename="sample.jpg"):
    gt_dir = tmp_path / "data" / "PaperBananaBench" / task_name
    gt_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (8, 8), color=(0, 0, 0))
    img.save(gt_dir / filename, format="JPEG")
    return filename


def _make_eval_data(tmp_path, task_name="diagram", gt_filename="sample.jpg"):
    return {
        "content": "methodology text",
        "visual_intent": "a pipeline diagram",
        "path_to_gt_image": gt_filename,
        "eval_image_field": f"target_{task_name}_desc0_base64_jpg",
        f"target_{task_name}_desc0_base64_jpg": _make_jpeg_b64(),
    }


# ===========================================================================
# _determine_tier_outcome
# ===========================================================================

class TestDetermineTierOutcome:

    def test_both_model(self):
        assert _determine_tier_outcome("Model", "Model") == "Model"

    def test_both_human(self):
        assert _determine_tier_outcome("Human", "Human") == "Human"

    def test_both_are_good_returns_tie(self):
        assert _determine_tier_outcome("Both are good", "Both are good") == "Tie"

    def test_both_are_bad_returns_tie(self):
        assert _determine_tier_outcome("Both are bad", "Both are bad") == "Tie"

    def test_model_plus_both_are_good(self):
        assert _determine_tier_outcome("Model", "Both are good") == "Model"

    def test_both_are_good_plus_model(self):
        assert _determine_tier_outcome("Both are good", "Model") == "Model"

    def test_model_plus_both_are_bad(self):
        assert _determine_tier_outcome("Model", "Both are bad") == "Model"

    def test_both_are_bad_plus_model(self):
        assert _determine_tier_outcome("Both are bad", "Model") == "Model"

    def test_human_plus_both_are_good(self):
        assert _determine_tier_outcome("Human", "Both are good") == "Human"

    def test_human_plus_both_are_bad(self):
        assert _determine_tier_outcome("Human", "Both are bad") == "Human"

    def test_both_are_good_plus_human(self):
        assert _determine_tier_outcome("Both are good", "Human") == "Human"

    def test_both_are_bad_plus_human(self):
        assert _determine_tier_outcome("Both are bad", "Human") == "Human"

    def test_model_vs_human_is_tie(self):
        assert _determine_tier_outcome("Model", "Human") == "Tie"

    def test_human_vs_model_is_tie(self):
        assert _determine_tier_outcome("Human", "Model") == "Tie"

    def test_both_are_good_vs_both_are_bad_is_tie(self):
        assert _determine_tier_outcome("Both are good", "Both are bad") == "Tie"

    def test_whitespace_is_stripped(self):
        assert _determine_tier_outcome("  Model  ", "  Model  ") == "Model"
        assert _determine_tier_outcome("  Human  ", "  Both are good  ") == "Human"


# ===========================================================================
# _try_regex_extract_winner
# ===========================================================================

class TestTryRegexExtractWinner:

    # --- Standard JSON format ---
    def test_standard_json_model(self):
        assert _try_regex_extract_winner('"winner": "Model"') == "Model"

    def test_standard_json_human(self):
        assert _try_regex_extract_winner('{"winner": "Human", "reasoning": "..."}') == "Human"

    def test_standard_json_both_are_good(self):
        assert _try_regex_extract_winner('"winner": "Both are good"') == "Both are good"

    def test_standard_json_both_are_bad(self):
        assert _try_regex_extract_winner('"winner": "Both are bad"') == "Both are bad"

    # --- Markdown bold format ---
    def test_markdown_bold_with_quotes_human(self):
        text = '**winner**: "Human"'
        assert _try_regex_extract_winner(text) == "Human"

    def test_markdown_bold_without_quotes_model(self):
        text = "**winner**: Model"
        result = _try_regex_extract_winner(text)
        assert result == "Model"

    def test_markdown_bold_both_are_good(self):
        text = "**winner**: Both are good"
        result = _try_regex_extract_winner(text)
        assert result == "Both are good"

    # --- Mixed / unquoted value ---
    def test_unquoted_value_in_json_key(self):
        text = '"winner": Both are good'
        result = _try_regex_extract_winner(text)
        assert result is not None and "Both are good" in result

    # --- Embedded in longer text ---
    def test_extracts_from_longer_reasoning_text(self):
        text = (
            'After careful comparison, I conclude that {"comparison_reasoning": '
            '"The model diagram is clearer", "winner": "Model"}'
        )
        assert _try_regex_extract_winner(text) == "Model"

    # --- No match ---
    def test_returns_none_when_no_winner_field(self):
        assert _try_regex_extract_winner('{"reasoning": "both look similar"}') is None

    def test_returns_none_on_empty_string(self):
        assert _try_regex_extract_winner("") is None

    def test_returns_none_on_plain_text_no_json(self):
        assert _try_regex_extract_winner("The model is better than human drawings.") is None

    # --- Case insensitivity ---
    def test_case_insensitive_winner_key(self):
        result = _try_regex_extract_winner('"WINNER": "Model"')
        assert result == "Model"


# ===========================================================================
# get_score_for_image_referenced — structural / routing tests
# ===========================================================================

@pytest.mark.asyncio
async def test_no_gt_path_sets_na_outcomes():
    """Missing path_to_gt_image → all dimension outcomes = 'N/A - No GT'."""
    data = {"content": "method", "visual_intent": "a diagram"}
    result = await get_score_for_image_referenced(data, task_name="diagram", model_name="")
    for dim in ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]:
        assert result[f"{dim}_outcome"] == "N/A - No GT"


@pytest.mark.asyncio
async def test_missing_eval_image_field_human_wins_all(tmp_path):
    """If eval_image_field key is absent from data, Human wins every dimension."""
    _write_gt_image(tmp_path)
    data = {
        "content": "method",
        "visual_intent": "a diagram",
        "path_to_gt_image": "sample.jpg",
        "eval_image_field": "target_diagram_desc0_base64_jpg",
        # key NOT set in data
    }
    result = await get_score_for_image_referenced(
        data, task_name="diagram", model_name="", work_dir=tmp_path
    )
    for dim in ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]:
        assert result[f"{dim}_outcome"] == "Human"


@pytest.mark.asyncio
async def test_tier1_decides_overall_when_both_agree_on_model(tmp_path):
    """Faithfulness=Model, Readability=Model → overall=Model (decided at Tier 1)."""
    _write_gt_image(tmp_path)
    data = _make_eval_data(tmp_path)

    dim_responses = {
        "faithfulness": {"winner": "Model", "comparison_reasoning": "Model wins"},
        "readability":  {"winner": "Model", "comparison_reasoning": "Model wins"},
        "conciseness":  {"winner": "Human", "comparison_reasoning": "Human wins"},
        "aesthetics":   {"winner": "Human", "comparison_reasoning": "Human wins"},
    }

    async def _mock_eval(task_name, eval_dim, *args, **kwargs):
        return eval_dim, dim_responses[eval_dim]

    with patch("utils.eval_toolkits._run_single_eval_ref", side_effect=_mock_eval):
        result = await get_score_for_image_referenced(
            data, task_name="diagram", model_name="gemini-test", work_dir=tmp_path
        )
    assert result["overall_outcome"] == "Model"
    assert "Tier 1" in result["overall_reasoning"]


@pytest.mark.asyncio
async def test_tier1_decides_overall_when_both_agree_on_human(tmp_path):
    _write_gt_image(tmp_path)
    data = _make_eval_data(tmp_path)

    dim_responses = {
        "faithfulness": {"winner": "Human", "comparison_reasoning": "..."},
        "readability":  {"winner": "Human", "comparison_reasoning": "..."},
        "conciseness":  {"winner": "Model", "comparison_reasoning": "..."},
        "aesthetics":   {"winner": "Model", "comparison_reasoning": "..."},
    }

    async def _mock_eval(task_name, eval_dim, *args, **kwargs):
        return eval_dim, dim_responses[eval_dim]

    with patch("utils.eval_toolkits._run_single_eval_ref", side_effect=_mock_eval):
        result = await get_score_for_image_referenced(
            data, task_name="diagram", model_name="gemini-test", work_dir=tmp_path
        )
    assert result["overall_outcome"] == "Human"


@pytest.mark.asyncio
async def test_tier1_tie_falls_back_to_tier2_model(tmp_path):
    """Tier 1: Model vs Human → Tie; Tier 2: Model+Model → overall=Model."""
    _write_gt_image(tmp_path)
    data = _make_eval_data(tmp_path)

    dim_responses = {
        "faithfulness": {"winner": "Model", "comparison_reasoning": "..."},
        "readability":  {"winner": "Human", "comparison_reasoning": "..."},
        "conciseness":  {"winner": "Model", "comparison_reasoning": "..."},
        "aesthetics":   {"winner": "Model", "comparison_reasoning": "..."},
    }

    async def _mock_eval(task_name, eval_dim, *args, **kwargs):
        return eval_dim, dim_responses[eval_dim]

    with patch("utils.eval_toolkits._run_single_eval_ref", side_effect=_mock_eval):
        result = await get_score_for_image_referenced(
            data, task_name="diagram", model_name="gemini-test", work_dir=tmp_path
        )
    assert result["overall_outcome"] == "Model"
    assert "Tier 2" in result["overall_reasoning"]


@pytest.mark.asyncio
async def test_tier1_tie_falls_back_to_tier2_human(tmp_path):
    """Tier 1: Human vs Model → Tie; Tier 2: Human+Human → overall=Human."""
    _write_gt_image(tmp_path)
    data = _make_eval_data(tmp_path)

    dim_responses = {
        "faithfulness": {"winner": "Human", "comparison_reasoning": "..."},
        "readability":  {"winner": "Model", "comparison_reasoning": "..."},
        "conciseness":  {"winner": "Human", "comparison_reasoning": "..."},
        "aesthetics":   {"winner": "Human", "comparison_reasoning": "..."},
    }

    async def _mock_eval(task_name, eval_dim, *args, **kwargs):
        return eval_dim, dim_responses[eval_dim]

    with patch("utils.eval_toolkits._run_single_eval_ref", side_effect=_mock_eval):
        result = await get_score_for_image_referenced(
            data, task_name="diagram", model_name="gemini-test", work_dir=tmp_path
        )
    assert result["overall_outcome"] == "Human"
    assert "Tier 2" in result["overall_reasoning"]


@pytest.mark.asyncio
async def test_all_neutral_outcomes_give_tie(tmp_path):
    """All dims neutral (Both are good/bad) → Tier 1 Tie → Tier 2 Tie → overall Tie."""
    _write_gt_image(tmp_path)
    data = _make_eval_data(tmp_path)

    dim_responses = {
        "faithfulness": {"winner": "Both are good", "comparison_reasoning": "..."},
        "readability":  {"winner": "Both are good", "comparison_reasoning": "..."},
        "conciseness":  {"winner": "Both are bad",  "comparison_reasoning": "..."},
        "aesthetics":   {"winner": "Both are bad",  "comparison_reasoning": "..."},
    }

    async def _mock_eval(task_name, eval_dim, *args, **kwargs):
        return eval_dim, dim_responses[eval_dim]

    with patch("utils.eval_toolkits._run_single_eval_ref", side_effect=_mock_eval):
        result = await get_score_for_image_referenced(
            data, task_name="diagram", model_name="gemini-test", work_dir=tmp_path
        )
    assert result["overall_outcome"] == "Tie"


@pytest.mark.asyncio
async def test_reasoning_stored_per_dimension(tmp_path):
    """Each dimension's comparison_reasoning must be saved in the result."""
    _write_gt_image(tmp_path)
    data = _make_eval_data(tmp_path)

    dim_responses = {
        "faithfulness": {"winner": "Model", "comparison_reasoning": "faithfulness reasoning"},
        "readability":  {"winner": "Model", "comparison_reasoning": "readability reasoning"},
        "conciseness":  {"winner": "Model", "comparison_reasoning": "conciseness reasoning"},
        "aesthetics":   {"winner": "Model", "comparison_reasoning": "aesthetics reasoning"},
    }

    async def _mock_eval(task_name, eval_dim, *args, **kwargs):
        return eval_dim, dim_responses[eval_dim]

    with patch("utils.eval_toolkits._run_single_eval_ref", side_effect=_mock_eval):
        result = await get_score_for_image_referenced(
            data, task_name="diagram", model_name="gemini-test", work_dir=tmp_path
        )
    for dim in ["faithfulness", "readability", "conciseness", "aesthetics"]:
        assert result[f"{dim}_reasoning"] == f"{dim} reasoning"


@pytest.mark.asyncio
async def test_plot_task_uses_correct_path(tmp_path):
    """Evaluation for plot task must resolve GT image from the plot subdirectory."""
    _write_gt_image(tmp_path, task_name="plot")
    data = {
        "content": "raw data",
        "visual_intent": "bar chart",
        "path_to_gt_image": "sample.jpg",
        "eval_image_field": "target_plot_desc0_base64_jpg",
        "target_plot_desc0_base64_jpg": _make_jpeg_b64(),
    }

    dim_responses = {
        "faithfulness": {"winner": "Model", "comparison_reasoning": "..."},
        "readability":  {"winner": "Model", "comparison_reasoning": "..."},
        "conciseness":  {"winner": "Model", "comparison_reasoning": "..."},
        "aesthetics":   {"winner": "Model", "comparison_reasoning": "..."},
    }

    async def _mock_eval(task_name, eval_dim, *args, **kwargs):
        return eval_dim, dim_responses[eval_dim]

    with patch("utils.eval_toolkits._run_single_eval_ref", side_effect=_mock_eval):
        result = await get_score_for_image_referenced(
            data, task_name="plot", model_name="gemini-test", work_dir=tmp_path
        )
    assert result["overall_outcome"] == "Model"


@pytest.mark.asyncio
async def test_tier1_one_model_one_neutral_gives_model(tmp_path):
    """Faithfulness=Model, Readability=Both are good → Tier 1 = Model."""
    _write_gt_image(tmp_path)
    data = _make_eval_data(tmp_path)

    dim_responses = {
        "faithfulness": {"winner": "Model",          "comparison_reasoning": "..."},
        "readability":  {"winner": "Both are good",  "comparison_reasoning": "..."},
        "conciseness":  {"winner": "Human",          "comparison_reasoning": "..."},
        "aesthetics":   {"winner": "Human",          "comparison_reasoning": "..."},
    }

    async def _mock_eval(task_name, eval_dim, *args, **kwargs):
        return eval_dim, dim_responses[eval_dim]

    with patch("utils.eval_toolkits._run_single_eval_ref", side_effect=_mock_eval):
        result = await get_score_for_image_referenced(
            data, task_name="diagram", model_name="gemini-test", work_dir=tmp_path
        )
    assert result["overall_outcome"] == "Model"
