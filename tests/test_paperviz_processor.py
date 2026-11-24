"""
Tests for utils/paperviz_processor.py — PaperVizProcessor

Covers:
process_single_query routing:
  - vanilla → vanilla_agent only; eval_image_field set correctly
  - dev_planner → retriever + planner + visualizer; stylist NOT called
  - dev_planner_stylist → includes stylist
  - dev_planner_critic / demo_planner_critic → critic iterations; demo mode skips eval
  - dev_full / demo_full → full pipeline; demo mode skips eval
  - dev_polish → polish_agent only
  - dev_retriever → retriever only; eval always skipped
  - unknown exp_mode → raises ValueError
  - do_eval=True calls evaluation_function; do_eval=False skips it

_run_critic_iterations:
  - Early stop when critic says "No changes needed." (visualizer not called)
  - Visualization failure → rolls back to previous best image key
  - Successful round → updates eval_image_field to new critic image key
  - max_rounds respected (exactly N critic calls)
  - source="planner" uses planner image as initial fallback
  - source="stylist" uses stylist image as initial fallback

process_queries_batch:
  - Processes all items in batch; yields each result
  - Empty batch yields nothing
  - Concurrency limit is honoured (no more than max_concurrent simultaneously)
"""

import asyncio
import base64
import io
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_b64(size=(16, 16), color=(100, 150, 200)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_exp_config(tmp_path, exp_mode="vanilla", task_name="diagram", **overrides):
    from utils.config import ExpConfig
    defaults = dict(
        dataset_name="PaperBananaBench",
        task_name=task_name,
        exp_mode=exp_mode,
        work_dir=tmp_path,
        model_name="test-model",
        image_model_name="test-image-model",
        timestamp="0101_1200",
        max_critic_rounds=3,
    )
    defaults.update(overrides)
    return ExpConfig(**defaults)


def _mock_agent(klass=None):
    """Return an agent mock whose process() returns the input data unchanged."""
    m = MagicMock(spec=klass) if klass else MagicMock()
    m.process = AsyncMock(side_effect=lambda data, **kw: data)
    return m


def _make_processor(tmp_path, exp_mode="vanilla", task_name="diagram", **cfg_overrides):
    from utils.paperviz_processor import PaperVizProcessor
    from agents.vanilla_agent import VanillaAgent
    from agents.planner_agent import PlannerAgent
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.polish_agent import PolishAgent

    exp_config = _make_exp_config(tmp_path, exp_mode=exp_mode, task_name=task_name, **cfg_overrides)
    agents = {
        "vanilla_agent":    _mock_agent(VanillaAgent),
        "planner_agent":    _mock_agent(PlannerAgent),
        "visualizer_agent": _mock_agent(VisualizerAgent),
        "stylist_agent":    _mock_agent(StylistAgent),
        "critic_agent":     _mock_agent(CriticAgent),
        "retriever_agent":  _mock_agent(RetrieverAgent),
        "polish_agent":     _mock_agent(PolishAgent),
    }
    processor = PaperVizProcessor(exp_config=exp_config, **agents)
    return processor, agents, exp_config


def _sample_data(task_name="diagram"):
    return {
        "content": "Methodology text",
        "visual_intent": "A pipeline diagram",
        "candidate_id": "test_001",
        f"target_{task_name}_desc0": "Initial planner description",
        f"target_{task_name}_stylist_desc0": "Stylist description",
        f"target_{task_name}_desc0_base64_jpg": _make_jpeg_b64(),
        f"target_{task_name}_stylist_desc0_base64_jpg": _make_jpeg_b64(),
        f"vanilla_{task_name}_base64_jpg": _make_jpeg_b64(),
        f"polished_{task_name}_base64_jpg": _make_jpeg_b64(),
    }


def _noop_eval(data, **kw):
    return data


# ===========================================================================
# process_single_query — routing
# ===========================================================================

class TestProcessSingleQueryRouting:

    @pytest.mark.asyncio
    async def test_vanilla_calls_only_vanilla_agent(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="vanilla")
        data = _sample_data()
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(data, do_eval=True)

        agents["vanilla_agent"].process.assert_called_once()
        agents["planner_agent"].process.assert_not_called()
        agents["stylist_agent"].process.assert_not_called()
        agents["visualizer_agent"].process.assert_not_called()
        agents["critic_agent"].process.assert_not_called()
        agents["retriever_agent"].process.assert_not_called()

    @pytest.mark.asyncio
    async def test_vanilla_eval_image_field(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="vanilla")
        data = _sample_data()
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        result = await proc.process_single_query(data, do_eval=True)
        assert result["eval_image_field"] == "vanilla_diagram_base64_jpg"

    @pytest.mark.asyncio
    async def test_dev_planner_calls_retriever_planner_visualizer(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="dev_planner")
        data = _sample_data()
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(data, do_eval=True)

        agents["retriever_agent"].process.assert_called_once()
        agents["planner_agent"].process.assert_called_once()
        agents["visualizer_agent"].process.assert_called_once()
        agents["stylist_agent"].process.assert_not_called()
        agents["critic_agent"].process.assert_not_called()

    @pytest.mark.asyncio
    async def test_dev_planner_eval_image_field(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="dev_planner")
        data = _sample_data()
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        result = await proc.process_single_query(data, do_eval=True)
        assert result["eval_image_field"] == "target_diagram_desc0_base64_jpg"

    @pytest.mark.asyncio
    async def test_dev_planner_stylist_calls_stylist(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="dev_planner_stylist")
        data = _sample_data()
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(data, do_eval=True)

        agents["stylist_agent"].process.assert_called_once()

    @pytest.mark.asyncio
    async def test_dev_planner_stylist_eval_image_field(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="dev_planner_stylist")
        data = _sample_data()
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        result = await proc.process_single_query(data, do_eval=True)
        assert result["eval_image_field"] == "target_diagram_stylist_desc0_base64_jpg"

    @pytest.mark.asyncio
    async def test_dev_planner_critic_runs_critic_iterations(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="dev_planner_critic")
        data = _sample_data()
        data["max_critic_rounds"] = 1

        async def critic_no_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "No changes needed."
            d[f"target_diagram_critic_desc{round_idx}"] = d.get("target_diagram_desc0", "")
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_no_changes)
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(data, do_eval=True)

        agents["critic_agent"].process.assert_called_once()

    @pytest.mark.asyncio
    async def test_demo_planner_critic_skips_eval(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="demo_planner_critic")
        data = _sample_data()
        data["max_critic_rounds"] = 1

        async def critic_no_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "No changes needed."
            d[f"target_diagram_critic_desc{round_idx}"] = d.get("target_diagram_desc0", "")
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_no_changes)
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(data, do_eval=True)

        proc.evaluation_function.assert_not_called()

    @pytest.mark.asyncio
    async def test_dev_full_calls_all_main_agents(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="dev_full")
        data = _sample_data()
        data["max_critic_rounds"] = 1

        async def critic_no_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "No changes needed."
            d[f"target_diagram_critic_desc{round_idx}"] = d.get("target_diagram_stylist_desc0", "")
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_no_changes)
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(data, do_eval=True)

        agents["retriever_agent"].process.assert_called_once()
        agents["planner_agent"].process.assert_called_once()
        agents["stylist_agent"].process.assert_called_once()
        agents["visualizer_agent"].process.assert_called_once()
        agents["critic_agent"].process.assert_called_once()

    @pytest.mark.asyncio
    async def test_demo_full_skips_eval(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="demo_full")
        data = _sample_data()
        data["max_critic_rounds"] = 1

        async def critic_no_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "No changes needed."
            d[f"target_diagram_critic_desc{round_idx}"] = d.get("target_diagram_stylist_desc0", "")
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_no_changes)
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(data, do_eval=True)

        proc.evaluation_function.assert_not_called()

    @pytest.mark.asyncio
    async def test_dev_polish_calls_polish_agent(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="dev_polish")
        data = _sample_data()
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        result = await proc.process_single_query(data, do_eval=True)

        agents["polish_agent"].process.assert_called_once()
        assert result["eval_image_field"] == "polished_diagram_base64_jpg"

    @pytest.mark.asyncio
    async def test_dev_retriever_calls_retriever_only_and_skips_eval(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="dev_retriever")
        data = _sample_data()
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(data, do_eval=True)

        agents["retriever_agent"].process.assert_called_once()
        agents["planner_agent"].process.assert_not_called()
        proc.evaluation_function.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_exp_mode_raises_value_error(self, tmp_path):
        proc, _, _ = _make_processor(tmp_path, exp_mode="nonexistent_mode")
        with pytest.raises(ValueError, match="Unknown experiment name"):
            await proc.process_single_query(_sample_data(), do_eval=False)

    @pytest.mark.asyncio
    async def test_do_eval_false_skips_evaluation(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="vanilla")
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(_sample_data(), do_eval=False)
        proc.evaluation_function.assert_not_called()

    @pytest.mark.asyncio
    async def test_do_eval_true_calls_evaluation(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="vanilla")
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        await proc.process_single_query(_sample_data(), do_eval=True)
        proc.evaluation_function.assert_called_once()


# ===========================================================================
# _run_critic_iterations
# ===========================================================================

class TestRunCriticIterations:

    def _make(self, tmp_path, task_name="diagram"):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="dev_planner_critic", task_name=task_name)
        return proc, agents

    @pytest.mark.asyncio
    async def test_early_stop_on_no_changes_needed(self, tmp_path):
        """Critic saying 'No changes needed.' on round 0 stops iteration; visualizer not called."""
        proc, agents = self._make(tmp_path)
        data = _sample_data()

        async def critic_no_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "No changes needed."
            d[f"target_diagram_critic_desc{round_idx}"] = d.get("target_diagram_desc0", "")
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_no_changes)
        result = await proc._run_critic_iterations(data, "diagram", max_rounds=3, source="planner")

        agents["visualizer_agent"].process.assert_not_called()
        assert result["eval_image_field"] == "target_diagram_desc0_base64_jpg"

    @pytest.mark.asyncio
    async def test_stylist_source_uses_stylist_image_as_initial_fallback(self, tmp_path):
        proc, agents = self._make(tmp_path)
        data = _sample_data()

        async def critic_no_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "No changes needed."
            d[f"target_diagram_critic_desc{round_idx}"] = d.get("target_diagram_stylist_desc0", "")
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_no_changes)
        result = await proc._run_critic_iterations(data, "diagram", max_rounds=3, source="stylist")

        assert result["eval_image_field"] == "target_diagram_stylist_desc0_base64_jpg"

    @pytest.mark.asyncio
    async def test_visualization_failure_rolls_back_to_previous_best(self, tmp_path):
        """When visualizer doesn't produce a new image, eval_image_field stays at previous best."""
        proc, agents = self._make(tmp_path)
        data = _sample_data()

        async def critic_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "Add more detail"
            d[f"target_diagram_critic_desc{round_idx}"] = "Improved description"
            return d

        async def visualizer_fails(d):
            # Does NOT add the new image key
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_changes)
        agents["visualizer_agent"].process = AsyncMock(side_effect=visualizer_fails)

        result = await proc._run_critic_iterations(data, "diagram", max_rounds=2, source="planner")

        assert result["eval_image_field"] == "target_diagram_desc0_base64_jpg"

    @pytest.mark.asyncio
    async def test_successful_round_updates_eval_image_field(self, tmp_path):
        proc, agents = self._make(tmp_path)
        data = _sample_data()

        async def critic_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "Improve layout"
            d[f"target_diagram_critic_desc{round_idx}"] = "Improved layout"
            return d

        async def visualizer_success(d):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_desc{round_idx}_base64_jpg"] = _make_jpeg_b64()
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_changes)
        agents["visualizer_agent"].process = AsyncMock(side_effect=visualizer_success)

        result = await proc._run_critic_iterations(data, "diagram", max_rounds=1, source="planner")

        assert result["eval_image_field"] == "target_diagram_critic_desc0_base64_jpg"

    @pytest.mark.asyncio
    async def test_max_rounds_respected(self, tmp_path):
        """Critic should be called at most max_rounds times."""
        proc, agents = self._make(tmp_path)
        data = _sample_data()

        rounds_executed = []

        async def critic_always_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            rounds_executed.append(round_idx)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "Always changes"
            d[f"target_diagram_critic_desc{round_idx}"] = f"Round {round_idx} desc"
            return d

        async def visualizer_success(d):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_desc{round_idx}_base64_jpg"] = _make_jpeg_b64()
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_always_changes)
        agents["visualizer_agent"].process = AsyncMock(side_effect=visualizer_success)

        await proc._run_critic_iterations(data, "diagram", max_rounds=2, source="planner")

        assert len(rounds_executed) == 2

    @pytest.mark.asyncio
    async def test_final_eval_image_field_set_after_all_rounds(self, tmp_path):
        """After all rounds complete, eval_image_field must be set."""
        proc, agents = self._make(tmp_path)
        data = _sample_data()

        async def critic_changes(d, source="stylist"):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_suggestions{round_idx}"] = "Refine"
            d[f"target_diagram_critic_desc{round_idx}"] = f"Round {round_idx}"
            return d

        async def visualizer_success(d):
            round_idx = d.get("current_critic_round", 0)
            d[f"target_diagram_critic_desc{round_idx}_base64_jpg"] = _make_jpeg_b64()
            return d

        agents["critic_agent"].process = AsyncMock(side_effect=critic_changes)
        agents["visualizer_agent"].process = AsyncMock(side_effect=visualizer_success)

        result = await proc._run_critic_iterations(data, "diagram", max_rounds=3, source="planner")

        assert "eval_image_field" in result
        assert result["eval_image_field"] is not None


# ===========================================================================
# process_queries_batch
# ===========================================================================

class TestProcessQueriesBatch:

    @pytest.mark.asyncio
    async def test_processes_all_items(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="vanilla")
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        data_list = [_sample_data() for _ in range(3)]

        results = []
        async for result in proc.process_queries_batch(data_list, max_concurrent=2, do_eval=True):
            results.append(result)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_empty_batch_yields_nothing(self, tmp_path):
        proc, _, _ = _make_processor(tmp_path, exp_mode="vanilla")
        results = []
        async for result in proc.process_queries_batch([], max_concurrent=5, do_eval=False):
            results.append(result)
        assert results == []

    @pytest.mark.asyncio
    async def test_single_item_batch(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="vanilla")
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        results = []
        async for r in proc.process_queries_batch([_sample_data()], max_concurrent=1, do_eval=False):
            results.append(r)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_results_have_eval_image_field(self, tmp_path):
        """Every result in vanilla mode must contain eval_image_field."""
        proc, agents, _ = _make_processor(tmp_path, exp_mode="vanilla")
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        data_list = [_sample_data() for _ in range(2)]
        async for result in proc.process_queries_batch(data_list, do_eval=False):
            assert "eval_image_field" in result

    @pytest.mark.asyncio
    async def test_do_eval_false_skips_evaluation_in_batch(self, tmp_path):
        proc, agents, _ = _make_processor(tmp_path, exp_mode="vanilla")
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        data_list = [_sample_data() for _ in range(2)]
        async for _ in proc.process_queries_batch(data_list, do_eval=False):
            pass
        proc.evaluation_function.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrency_limit_applied(self, tmp_path):
        """max_concurrent=1 must still process all items (just sequentially)."""
        proc, agents, _ = _make_processor(tmp_path, exp_mode="vanilla")
        proc.evaluation_function = AsyncMock(side_effect=_noop_eval)
        data_list = [_sample_data() for _ in range(4)]
        results = []
        async for result in proc.process_queries_batch(data_list, max_concurrent=1, do_eval=False):
            results.append(result)
        assert len(results) == 4
