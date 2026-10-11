"""Starter tests for repository initialization (Phase 0)."""

from pathlib import Path
from cloud_rainfall.config import PROJECT_ROOT, get_paths_config, get_cloud_model_config, get_rainfall_model_config


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "AGENTS.md").exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert (PROJECT_ROOT / "requirements.txt").exists()


def test_configs_loadable():
    paths_cfg = get_paths_config()
    assert "data" in paths_cfg
    assert "artifacts" in paths_cfg

    cloud_cfg = get_cloud_model_config()
    assert "model" in cloud_cfg
    assert "backbone" in cloud_cfg["model"]

    rain_cfg = get_rainfall_model_config()
    assert "bins" in rain_cfg
    assert len(rain_cfg["bins"]) == 4
