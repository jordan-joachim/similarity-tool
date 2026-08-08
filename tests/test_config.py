"""Tests for config loading/saving, defaults, and cache-directory creation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from similarity_tool import config as config_mod
from similarity_tool.config import (
    Config,
    config_dir,
    config_path,
    create_cache_dir,
    ensure_config_file,
    load_config,
    save_config,
)


class TestDefaults:
    def test_defaults_match_architecture(self):
        cfg = Config()
        assert cfg.photo_root == "/run/media/joachim/LinStorage/Media/Sammlung/Bilder"
        assert cfg.trash_root == "~/.local/share/similarity-tool/trash"
        assert cfg.cache_path == "~/.cache/similarity-tool/hashes.sqlite3"
        assert cfg.file_extensions == [".jpg", ".jpeg"]
        assert cfg.hash_algorithms == ["phash", "dhash"]
        assert cfg.phash_threshold == 8
        assert cfg.dhash_threshold == 10
        assert cfg.ai_refinement is False
        assert cfg.ai_model == "openai/clip-vit-base-patch32"
        assert cfg.ai_similarity_threshold == 0.85
        assert cfg.blur_enabled is True
        assert cfg.blur_threshold_percentile == 10.0
        assert cfg.blur_min_absolute == 100.0

    def test_defaults_are_independent_copies(self):
        cfg = Config()
        cfg.file_extensions.append(".png")
        assert Config().file_extensions == [".jpg", ".jpeg"]


class TestFirstLaunch:
    def test_config_dir_and_file_created_on_first_launch(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        assert not config_dir().exists()
        path = ensure_config_file()
        assert path == config_path()
        assert config_dir().is_dir()
        data = json.loads(path.read_text())
        assert data["photo_root"] == Config().photo_root
        assert data["blur_threshold_percentile"] == 10.0
        assert data["phash_threshold"] == 8

    def test_existing_config_not_overwritten(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"phash_threshold": 3}))
        ensure_config_file(path)
        assert json.loads(path.read_text()) == {"phash_threshold": 3}

    def test_cache_dir_created_before_first_write(self, tmp_path):
        cfg = Config(cache_path=str(tmp_path / "cache" / "hashes.sqlite3"))
        assert not (tmp_path / "cache").exists()
        cache = create_cache_dir(cfg)
        assert (tmp_path / "cache").is_dir()
        assert cache == tmp_path / "cache" / "hashes.sqlite3"

    def test_cache_dir_created_from_default_path(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        cfg = Config()
        cache = create_cache_dir(cfg)
        assert cache.parent == home / ".cache" / "similarity-tool"
        assert cache.parent.is_dir()


class TestLoad:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nope.json")
        assert cfg.phash_threshold == 8
        assert cfg.blur_enabled is True

    def test_custom_values_are_loaded(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"phash_threshold": 3, "blur_enabled": False}))
        cfg = load_config(path)
        assert cfg.phash_threshold == 3
        assert cfg.blur_enabled is False

    def test_missing_fields_use_documented_defaults(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"phash_threshold": 3}))
        cfg = load_config(path)
        assert cfg.phash_threshold == 3
        assert cfg.dhash_threshold == 10
        assert cfg.blur_threshold_percentile == 10.0
        assert cfg.blur_min_absolute == 100.0
        assert cfg.ai_refinement is False
        assert cfg.file_extensions == [".jpg", ".jpeg"]

    def test_malformed_json_logs_error_and_falls_back(self, tmp_path, caplog):
        path = tmp_path / "config.json"
        path.write_text("{not json!!")
        with caplog.at_level(logging.ERROR, logger="similarity_tool.config"):
            cfg = load_config(path)
        assert cfg.phash_threshold == 8
        assert cfg.blur_threshold_percentile == 10.0
        assert any("Could not read" in record.message for record in caplog.records)

    def test_non_object_json_logs_error_and_falls_back(self, tmp_path, caplog):
        path = tmp_path / "config.json"
        path.write_text("[1, 2, 3]")
        with caplog.at_level(logging.ERROR, logger="similarity_tool.config"):
            cfg = load_config(path)
        assert cfg.phash_threshold == 8
        assert any("does not contain a JSON object" in record.message for record in caplog.records)

    def test_wrong_type_logs_error_and_uses_default(self, tmp_path, caplog):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"phash_threshold": "many", "blur_enabled": 1}))
        with caplog.at_level(logging.ERROR, logger="similarity_tool.config"):
            cfg = load_config(path)
        assert cfg.phash_threshold == 8
        assert cfg.blur_enabled is True
        assert any("unsupported value" in record.message for record in caplog.records)

    def test_partial_wrong_type_keeps_valid_fields(self, tmp_path, caplog):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"phash_threshold": 3, "blur_enabled": "yes"}))
        with caplog.at_level(logging.ERROR, logger="similarity_tool.config"):
            cfg = load_config(path)
        assert cfg.phash_threshold == 3
        assert cfg.blur_enabled is True


class TestSave:
    def test_save_roundtrip(self, tmp_path):
        path = tmp_path / "config.json"
        cfg = Config(phash_threshold=3, blur_enabled=False, file_extensions=[".jpg"])
        result = save_config(cfg, path)
        assert result == path
        loaded = load_config(path)
        assert loaded.phash_threshold == 3
        assert loaded.blur_enabled is False
        assert loaded.file_extensions == [".jpg"]

    def test_save_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "config.json"
        save_config(Config(), path)
        assert path.is_file()
        assert json.loads(path.read_text())["phash_threshold"] == 8

    def test_save_uses_default_path(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        save_config(Config(phash_threshold=5))
        assert config_path().is_file()
        assert json.loads(config_path().read_text())["phash_threshold"] == 5
