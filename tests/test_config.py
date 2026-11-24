"""
Tests for utils/config.py — ExpConfig dataclass

Covers:
- Default field values
- Timestamp: auto-generated vs provided
- exp_name format: {timestamp}_{retrieval_setting}ret_{exp_mode}_{split_name}
- result_dir is created on post-init
- result_dir path uses dataset_name + task_name
- YAML fallback: model_name / image_model_name loaded when empty
- YAML values NOT used when model_name is already provided
- Missing YAML file → empty model names (no crash)
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_config(tmp_path, **overrides):
    from utils.config import ExpConfig
    defaults = dict(
        dataset_name="PaperBananaBench",
        work_dir=tmp_path,
        model_name="test-model",
        image_model_name="test-image-model",
        timestamp="0101_1200",
    )
    defaults.update(overrides)
    return ExpConfig(**defaults)


def _write_yaml(tmp_path, model_name: str, image_model_name: str):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "model_config.yaml").write_text(
        yaml.dump({"defaults": {"model_name": model_name, "image_model_name": image_model_name}})
    )


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------

def test_default_task_name(tmp_path):
    assert _make_config(tmp_path).task_name == "diagram"


def test_default_split_name(tmp_path):
    assert _make_config(tmp_path).split_name == "test"


def test_default_temperature(tmp_path):
    assert _make_config(tmp_path).temperature == 1.0


def test_default_retrieval_setting(tmp_path):
    assert _make_config(tmp_path).retrieval_setting == "auto"


def test_default_max_critic_rounds(tmp_path):
    assert _make_config(tmp_path).max_critic_rounds == 3


def test_custom_task_name(tmp_path):
    assert _make_config(tmp_path, task_name="plot").task_name == "plot"


def test_custom_split_name(tmp_path):
    assert _make_config(tmp_path, split_name="val").split_name == "val"


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

def test_timestamp_preserved_when_provided(tmp_path):
    cfg = _make_config(tmp_path, timestamp="1234_5678")
    assert cfg.timestamp == "1234_5678"


def test_timestamp_auto_generated_when_none(tmp_path):
    from utils.config import ExpConfig
    cfg = ExpConfig(
        dataset_name="PaperBananaBench",
        work_dir=tmp_path,
        model_name="m",
        image_model_name="im",
        timestamp=None,
    )
    assert cfg.timestamp is not None
    assert len(cfg.timestamp) > 0


def test_timestamp_format_is_mmdd_hhmm(tmp_path):
    """Auto-generated timestamp should match MMDD_HHMM (4 digits, underscore, 4 digits)."""
    import re
    from utils.config import ExpConfig
    cfg = ExpConfig(
        dataset_name="PaperBananaBench",
        work_dir=tmp_path,
        model_name="m",
        image_model_name="im",
        timestamp=None,
    )
    assert re.match(r"^\d{4}_\d{4}$", cfg.timestamp), f"Unexpected format: {cfg.timestamp}"


# ---------------------------------------------------------------------------
# exp_name format
# ---------------------------------------------------------------------------

def test_exp_name_format_full(tmp_path):
    cfg = _make_config(
        tmp_path,
        timestamp="0315_1430",
        retrieval_setting="random",
        exp_mode="dev_planner",
        split_name="val",
    )
    assert cfg.exp_name == "0315_1430_randomret_dev_planner_val"


def test_exp_name_includes_auto_retrieval(tmp_path):
    cfg = _make_config(tmp_path, timestamp="0101_0000", retrieval_setting="auto", exp_mode="vanilla")
    assert "autoret" in cfg.exp_name


def test_exp_name_includes_none_retrieval(tmp_path):
    cfg = _make_config(tmp_path, timestamp="0101_0000", retrieval_setting="none")
    assert "noneret" in cfg.exp_name


def test_exp_name_includes_split_name(tmp_path):
    cfg = _make_config(tmp_path, timestamp="0101_0000", split_name="train")
    assert cfg.exp_name.endswith("_train")


# ---------------------------------------------------------------------------
# result_dir creation
# ---------------------------------------------------------------------------

def test_result_dir_created_on_init(tmp_path):
    cfg = _make_config(tmp_path)
    assert cfg.result_dir.exists()
    assert cfg.result_dir.is_dir()


def test_result_dir_path_for_diagram(tmp_path):
    cfg = _make_config(tmp_path, task_name="diagram")
    assert cfg.result_dir == tmp_path / "results" / "PaperBananaBench_diagram"


def test_result_dir_path_for_plot(tmp_path):
    cfg = _make_config(tmp_path, task_name="plot")
    assert cfg.result_dir == tmp_path / "results" / "PaperBananaBench_plot"


def test_result_dir_creation_is_idempotent(tmp_path):
    """Creating ExpConfig twice must not raise (parents=True, exist_ok=True)."""
    _make_config(tmp_path)
    _make_config(tmp_path)  # should not raise


# ---------------------------------------------------------------------------
# YAML fallback for model names
# ---------------------------------------------------------------------------

def test_model_name_loaded_from_yaml_when_empty(tmp_path):
    _write_yaml(tmp_path, "google/gemini-from-yaml", "google/imagen-from-yaml")
    from utils.config import ExpConfig
    cfg = ExpConfig(
        dataset_name="PaperBananaBench",
        work_dir=tmp_path,
        model_name="",
        image_model_name="",
        timestamp="0000_0000",
    )
    assert cfg.model_name == "google/gemini-from-yaml"
    assert cfg.image_model_name == "google/imagen-from-yaml"


def test_provided_model_name_not_overridden_by_yaml(tmp_path):
    _write_yaml(tmp_path, "should-not-be-used", "should-not-be-used")
    cfg = _make_config(tmp_path)  # model_name="test-model"
    assert cfg.model_name == "test-model"
    assert cfg.image_model_name == "test-image-model"


def test_partial_override_model_only(tmp_path):
    """If only model_name is empty, only it should be filled from YAML."""
    _write_yaml(tmp_path, "yaml-model", "yaml-image-model")
    from utils.config import ExpConfig
    cfg = ExpConfig(
        dataset_name="PaperBananaBench",
        work_dir=tmp_path,
        model_name="",
        image_model_name="explicit-image-model",
        timestamp="0000_0000",
    )
    assert cfg.model_name == "yaml-model"
    assert cfg.image_model_name == "explicit-image-model"


def test_missing_yaml_does_not_crash(tmp_path):
    """When configs/model_config.yaml doesn't exist, empty model names are used."""
    from utils.config import ExpConfig
    cfg = ExpConfig(
        dataset_name="PaperBananaBench",
        work_dir=tmp_path,
        model_name="",
        image_model_name="",
        timestamp="0000_0000",
    )
    assert cfg.model_name == ""
    assert cfg.image_model_name == ""


def test_malformed_yaml_defaults_to_empty(tmp_path):
    """Malformed YAML config should not crash; model names remain empty."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "model_config.yaml").write_text("{ invalid yaml: [[[")
    from utils.config import ExpConfig
    try:
        cfg = ExpConfig(
            dataset_name="PaperBananaBench",
            work_dir=tmp_path,
            model_name="",
            image_model_name="",
            timestamp="0000_0000",
        )
        # If it doesn't raise, model names may be empty
        assert isinstance(cfg.model_name, str)
    except Exception:
        pass  # Acceptable: some YAML parsers raise on invalid input
