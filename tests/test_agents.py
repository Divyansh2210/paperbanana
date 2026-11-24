"""
Tests for agents/critic_agent.py and agents/retriever_agent.py

CriticAgent covers:
- Round 0 with source="planner" uses planner description key
- Round 0 with source="stylist" uses stylist description key
- Round > 0 uses previous critic output key
- Invalid source raises ValueError
- "No changes needed." preserves original description
- Missing/empty image falls back to text-only critique (SYSTEM NOTICE)
- dict/list content is JSON-serialized before sending
- JSON parse failure produces default empty result without crashing
- Plot task uses plot-specific keys

RetrieverAgent covers:
- retrieval_setting="none" returns empty lists immediately
- Unknown retrieval_setting raises ValueError
- auto / random fall back to "none" when ref file is missing
- manual falls back to "none" when agent_selected_12.json is missing
- random returns up to 10 IDs from a pool; respects pool size < 10
- auto calls model, parses top10_diagrams (diagram) / top10_plots (plot)
- _parse_retrieval_result: valid JSON diagram, valid JSON plot, bad JSON → [],
  unknown task_name raises
"""

import base64
import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

PATCH_GEMINI = "utils.generation_utils.call_gemini_with_retry_async"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_exp_config(tmp_path, task_name="diagram", exp_mode="dev_planner_critic", **overrides):
    from utils.config import ExpConfig
    defaults = dict(
        dataset_name="PaperBananaBench",
        task_name=task_name,
        exp_mode=exp_mode,
        work_dir=tmp_path,
        model_name="test-model",
        image_model_name="test-image-model",
        timestamp="0101_1200",
    )
    defaults.update(overrides)
    return ExpConfig(**defaults)


def _make_jpeg_b64(size=(16, 16), color=(100, 150, 200)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _write_ref_json(tmp_path, task_name="diagram", n: int = 5):
    ref_dir = tmp_path / "data" / "PaperBananaBench" / task_name
    ref_dir.mkdir(parents=True, exist_ok=True)
    items = [
        {"id": f"ref_{i}", "visual_intent": f"intent {i}", "content": f"content {i}"}
        for i in range(n)
    ]
    (ref_dir / "ref.json").write_text(json.dumps(items))
    return ref_dir


# ===========================================================================
# CriticAgent
# ===========================================================================

class TestCriticAgent:

    def _agent(self, tmp_path, task_name="diagram"):
        from agents.critic_agent import CriticAgent
        cfg = _make_exp_config(tmp_path, task_name=task_name)
        return CriticAgent(model_name=cfg.model_name, system_prompt="", exp_config=cfg)

    def _base_data(self, task_name="diagram", round_idx=0):
        return {
            "content": "Methodology text.",
            "visual_intent": "A pipeline diagram.",
            "current_critic_round": round_idx,
            f"target_{task_name}_desc0": "Planner description.",
            f"target_{task_name}_stylist_desc0": "Stylist description.",
            f"target_{task_name}_desc0_base64_jpg": _make_jpeg_b64(),
            f"target_{task_name}_stylist_desc0_base64_jpg": _make_jpeg_b64(),
        }

    # --- source routing ---

    @pytest.mark.asyncio
    async def test_round0_planner_source_stores_in_round0_key(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data()
        resp = json.dumps({"critic_suggestions": "Add arrows", "revised_description": "With arrows"})
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[resp]):
            result = await agent.process(data, source="planner")
        assert result["target_diagram_critic_desc0"] == "With arrows"
        assert result["target_diagram_critic_suggestions0"] == "Add arrows"

    @pytest.mark.asyncio
    async def test_round0_stylist_source_stores_in_round0_key(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data()
        resp = json.dumps({"critic_suggestions": "Fix colors", "revised_description": "Better colors"})
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[resp]):
            result = await agent.process(data, source="stylist")
        assert result["target_diagram_critic_desc0"] == "Better colors"

    @pytest.mark.asyncio
    async def test_invalid_source_raises_value_error(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data()
        with pytest.raises(ValueError, match="Invalid source"):
            await agent.process(data, source="wrong_source")

    @pytest.mark.asyncio
    async def test_round1_uses_previous_critic_output_key(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data(round_idx=1)
        data["target_diagram_critic_desc0"] = "Round 0 output"
        data["target_diagram_critic_desc0_base64_jpg"] = _make_jpeg_b64()
        resp = json.dumps({"critic_suggestions": "More detail", "revised_description": "Round 1 output"})
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[resp]):
            result = await agent.process(data, source="stylist")
        assert result["target_diagram_critic_desc1"] == "Round 1 output"

    # --- "No changes needed." handling ---

    @pytest.mark.asyncio
    async def test_no_changes_needed_preserves_original_description(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data()
        original_desc = data["target_diagram_desc0"]
        resp = json.dumps({
            "critic_suggestions": "No changes needed.",
            "revised_description": "No changes needed.",
        })
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[resp]):
            result = await agent.process(data, source="planner")
        assert result["target_diagram_critic_desc0"] == original_desc

    @pytest.mark.asyncio
    async def test_no_changes_with_whitespace_preserves_original(self, tmp_path):
        """'No changes needed.' with surrounding whitespace must be treated the same."""
        agent = self._agent(tmp_path)
        data = self._base_data()
        original_desc = data["target_diagram_desc0"]
        resp = json.dumps({
            "critic_suggestions": "No changes needed.",
            "revised_description": "  No changes needed.  ",
        })
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[resp]):
            result = await agent.process(data, source="planner")
        assert result["target_diagram_critic_desc0"] == original_desc

    # --- Image presence / absence ---

    @pytest.mark.asyncio
    async def test_short_image_triggers_text_only_mode(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data()
        data["target_diagram_desc0_base64_jpg"] = ""  # empty → text-only
        resp = json.dumps({"critic_suggestions": "ok", "revised_description": "ok"})
        captured = []

        async def _capture(*args, **kwargs):
            captured.append(kwargs.get("contents") or args[1])
            return [resp]

        with patch(PATCH_GEMINI, side_effect=_capture):
            await agent.process(data, source="planner")

        contents = captured[0]
        text_parts = [c["text"] for c in contents if c.get("type") == "text"]
        assert any("SYSTEM NOTICE" in t for t in text_parts)

    @pytest.mark.asyncio
    async def test_valid_image_included_in_contents(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data()
        resp = json.dumps({"critic_suggestions": "ok", "revised_description": "ok"})
        captured = []

        async def _capture(*args, **kwargs):
            captured.append(kwargs.get("contents") or args[1])
            return [resp]

        with patch(PATCH_GEMINI, side_effect=_capture):
            await agent.process(data, source="planner")

        contents = captured[0]
        image_parts = [c for c in contents if c.get("type") == "image"]
        assert len(image_parts) == 1

    # --- Content serialization ---

    @pytest.mark.asyncio
    async def test_dict_content_does_not_crash(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data()
        data["content"] = {"methodology": "Transformer-based model", "steps": [1, 2, 3]}
        resp = json.dumps({"critic_suggestions": "ok", "revised_description": "ok"})
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[resp]):
            result = await agent.process(data, source="planner")
        assert "target_diagram_critic_desc0" in result

    @pytest.mark.asyncio
    async def test_list_content_does_not_crash(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data()
        data["content"] = ["step A", "step B", "step C"]
        resp = json.dumps({"critic_suggestions": "ok", "revised_description": "ok"})
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[resp]):
            result = await agent.process(data, source="planner")
        assert "target_diagram_critic_desc0" in result

    # --- JSON parse failure ---

    @pytest.mark.asyncio
    async def test_malformed_json_produces_default_keys(self, tmp_path):
        agent = self._agent(tmp_path)
        data = self._base_data()
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=["NOT VALID JSON {{{["]):
            result = await agent.process(data, source="planner")
        assert "target_diagram_critic_desc0" in result
        assert "target_diagram_critic_suggestions0" in result

    # --- Plot task ---

    @pytest.mark.asyncio
    async def test_plot_task_uses_plot_keys(self, tmp_path):
        agent = self._agent(tmp_path, task_name="plot")
        data = {
            "content": "Raw CSV data",
            "visual_intent": "A bar chart.",
            "current_critic_round": 0,
            "target_plot_desc0": "Plot planner description",
            "target_plot_stylist_desc0": "Plot stylist description",
            "target_plot_desc0_base64_jpg": _make_jpeg_b64(),
            "target_plot_stylist_desc0_base64_jpg": _make_jpeg_b64(),
        }
        resp = json.dumps({"critic_suggestions": "Fix axis", "revised_description": "Fixed axis"})
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[resp]):
            result = await agent.process(data, source="planner")
        assert "target_plot_critic_desc0" in result
        assert "target_plot_critic_suggestions0" in result


# ===========================================================================
# RetrieverAgent
# ===========================================================================

class TestRetrieverAgent:

    def _agent(self, tmp_path, task_name="diagram"):
        from agents.retriever_agent import RetrieverAgent
        cfg = _make_exp_config(tmp_path, task_name=task_name)
        return RetrieverAgent(model_name=cfg.model_name, system_prompt="", exp_config=cfg)

    def _base_data(self):
        return {"content": "Methodology text", "visual_intent": "A diagram."}

    # --- retrieval_setting="none" ---

    @pytest.mark.asyncio
    async def test_none_returns_empty_lists(self, tmp_path):
        result = await self._agent(tmp_path).process(self._base_data(), retrieval_setting="none")
        assert result["top10_references"] == []
        assert result["retrieved_examples"] == []

    # --- Unknown retrieval_setting ---

    @pytest.mark.asyncio
    async def test_unknown_setting_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown retrieval_setting"):
            await self._agent(tmp_path).process(self._base_data(), retrieval_setting="mystery")

    # --- Graceful fallback when files are missing ---

    @pytest.mark.asyncio
    async def test_auto_fallback_to_none_when_ref_missing(self, tmp_path):
        result = await self._agent(tmp_path).process(self._base_data(), retrieval_setting="auto")
        assert result["top10_references"] == []

    @pytest.mark.asyncio
    async def test_random_fallback_to_none_when_ref_missing(self, tmp_path):
        result = await self._agent(tmp_path).process(self._base_data(), retrieval_setting="random")
        assert result["top10_references"] == []

    @pytest.mark.asyncio
    async def test_manual_fallback_to_none_when_file_missing(self, tmp_path):
        result = await self._agent(tmp_path).process(self._base_data(), retrieval_setting="manual")
        assert result["top10_references"] == []

    # --- retrieval_setting="random" ---

    @pytest.mark.asyncio
    async def test_random_returns_at_most_10(self, tmp_path):
        _write_ref_json(tmp_path, n=20)
        result = await self._agent(tmp_path).process(self._base_data(), retrieval_setting="random")
        assert len(result["top10_references"]) <= 10

    @pytest.mark.asyncio
    async def test_random_with_small_pool_returns_all(self, tmp_path):
        _write_ref_json(tmp_path, n=3)
        result = await self._agent(tmp_path).process(self._base_data(), retrieval_setting="random")
        assert len(result["top10_references"]) == 3

    @pytest.mark.asyncio
    async def test_random_ids_are_valid(self, tmp_path):
        """All returned IDs must come from the reference pool."""
        _write_ref_json(tmp_path, n=10)
        valid_ids = {f"ref_{i}" for i in range(10)}
        result = await self._agent(tmp_path).process(self._base_data(), retrieval_setting="random")
        for ref_id in result["top10_references"]:
            assert ref_id in valid_ids

    @pytest.mark.asyncio
    async def test_random_empty_pool_returns_empty(self, tmp_path):
        _write_ref_json(tmp_path, n=0)
        result = await self._agent(tmp_path).process(self._base_data(), retrieval_setting="random")
        assert result["top10_references"] == []

    # --- retrieval_setting="auto" ---

    @pytest.mark.asyncio
    async def test_auto_diagram_parses_top10_diagrams(self, tmp_path):
        _write_ref_json(tmp_path, n=5)
        agent = self._agent(tmp_path)
        model_resp = json.dumps({"top10_diagrams": ["ref_0", "ref_2", "ref_4"]})
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[model_resp]):
            result = await agent.process(self._base_data(), retrieval_setting="auto")
        assert result["top10_references"] == ["ref_0", "ref_2", "ref_4"]

    @pytest.mark.asyncio
    async def test_auto_plot_parses_top10_plots(self, tmp_path):
        _write_ref_json(tmp_path, task_name="plot", n=5)
        agent = self._agent(tmp_path, task_name="plot")
        model_resp = json.dumps({"top10_plots": ["ref_1", "ref_3"]})
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=[model_resp]):
            result = await agent.process(
                {"content": "data", "visual_intent": "bar chart"}, retrieval_setting="auto"
            )
        assert result["top10_references"] == ["ref_1", "ref_3"]

    @pytest.mark.asyncio
    async def test_auto_model_failure_returns_empty(self, tmp_path):
        """If model returns unparseable response, auto retrieval returns empty list."""
        _write_ref_json(tmp_path, n=5)
        agent = self._agent(tmp_path)
        with patch(PATCH_GEMINI, new_callable=AsyncMock, return_value=["NOT_JSON"]):
            result = await agent.process(self._base_data(), retrieval_setting="auto")
        assert result["top10_references"] == []

    # --- _parse_retrieval_result ---

    def test_parse_diagram_result(self, tmp_path):
        agent = self._agent(tmp_path)
        raw = json.dumps({"top10_diagrams": ["ref_1", "ref_2", "ref_7"]})
        assert agent._parse_retrieval_result(raw, "diagram") == ["ref_1", "ref_2", "ref_7"]

    def test_parse_plot_result(self, tmp_path):
        agent = self._agent(tmp_path)
        raw = json.dumps({"top10_plots": ["ref_5", "ref_9"]})
        assert agent._parse_retrieval_result(raw, "plot") == ["ref_5", "ref_9"]

    def test_parse_empty_list_for_bad_json(self, tmp_path):
        agent = self._agent(tmp_path)
        assert agent._parse_retrieval_result("DEFINITELY NOT JSON", "diagram") == []

    def test_parse_empty_list_when_key_missing(self, tmp_path):
        agent = self._agent(tmp_path)
        raw = json.dumps({"wrong_key": ["ref_1"]})
        assert agent._parse_retrieval_result(raw, "diagram") == []

    def test_parse_unknown_task_returns_empty(self, tmp_path):
        """Unknown task_name triggers a ValueError that is caught internally → empty list."""
        agent = self._agent(tmp_path)
        result = agent._parse_retrieval_result('{"top10_diagrams": []}', "unknown_task")
        assert result == []

    def test_parse_json_with_markdown_fences(self, tmp_path):
        """Response wrapped in ```json ... ``` should still parse (via json_repair)."""
        agent = self._agent(tmp_path)
        raw = '```json\n{"top10_diagrams": ["ref_3", "ref_9"]}\n```'
        result = agent._parse_retrieval_result(raw, "diagram")
        assert result == ["ref_3", "ref_9"]
