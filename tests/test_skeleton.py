"""Skeleton tests: package metadata, config, models, and entry point."""

from __future__ import annotations

import json

import pytest

from similarity_tool import __version__
from similarity_tool import config as config_mod
from similarity_tool.config import Config, ensure_config_file, load_config
from similarity_tool.models import DeletionLog, PhotoFile


class TestPackageMetadata:
    def test_version_is_defined(self):
        assert isinstance(__version__, str)
        assert __version__.count(".") >= 1

    def test_config_defaults_match_architecture(self):
        cfg = Config()
        assert cfg.photo_root == config_mod.DEFAULT_PHOTO_ROOT
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


class TestConfigLoadSave:
    def test_ensure_config_creates_default_file(self, tmp_path):
        path = tmp_path / "config.json"
        result = ensure_config_file(path)
        assert result == path
        data = json.loads(path.read_text())
        assert data["photo_root"].startswith("/")
        assert data["blur_threshold_percentile"] == 10.0
        assert data["phash_threshold"] == 8

    def test_ensure_config_does_not_overwrite_existing(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"phash_threshold": 3}))
        ensure_config_file(path)
        data = json.loads(path.read_text())
        assert data == {"phash_threshold": 3}

    def test_load_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nope.json")
        assert cfg.phash_threshold == 8
        assert cfg.blur_enabled is True

    def test_load_custom_values(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"phash_threshold": 3, "blur_enabled": False}))
        cfg = load_config(path)
        assert cfg.phash_threshold == 3
        assert cfg.blur_enabled is False

    def test_malformed_json_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{not json!!")
        cfg = load_config(path)
        assert cfg.phash_threshold == 8
        assert cfg.blur_threshold_percentile == 10.0

    def test_wrong_type_falls_back_to_default(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"phash_threshold": "many", "blur_enabled": 1}))
        cfg = load_config(path)
        assert cfg.phash_threshold == 8
        assert cfg.blur_enabled is True


class TestModels:
    def test_photo_file_name(self):
        photo = PhotoFile(path=__import__("pathlib").Path("/a/b.jpg"), relative_path="2024/05/b.jpg", size=10, mtime=1.0)
        assert photo.name == "b.jpg"

    def test_deletion_log_roundtrip(self, tmp_path):
        from datetime import datetime

        log = DeletionLog()
        log.add(
            original_path=__import__("pathlib").Path("/archive/2024/05/x.jpg"),
            trash_path=__import__("pathlib").Path("/trash/2024-05-17/uuid/2024/05/x.jpg"),
            mode="similarity",
            timestamp=datetime(2024, 5, 17, 12, 0, 0),
            file_size=1234,
        )
        restored = DeletionLog.from_dict(log.to_dict())
        assert restored.entries[0].mode == "similarity"
        assert restored.entries[0].file_size == 1234
        assert restored.entries[0].original_path.startswith("/")


class TestEntryPoint:
    def test_main_help_exits_zero(self):
        from similarity_tool import __main__

        with pytest.raises(SystemExit) as excinfo:
            __main__.main(["--help"])
        assert excinfo.value.code == 0
