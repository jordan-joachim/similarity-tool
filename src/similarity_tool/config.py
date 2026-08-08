"""Configuration persistence for the Similarity Tool.

Settings live in ``~/.config/similarity-tool/config.json`` and are created on
first use with the documented defaults. A malformed or partial config falls
back to defaults rather than crashing.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_DIR = "~/.config/similarity-tool"
CONFIG_FILE = "config.json"

DEFAULT_PHOTO_ROOT = "/run/media/joachim/LinStorage/Media/Sammlung/Bilder"
DEFAULT_TRASH_ROOT = "~/.local/share/similarity-tool/trash"
DEFAULT_CACHE_PATH = "~/.cache/similarity-tool/hashes.sqlite3"

DEFAULT_FILE_EXTENSIONS = [".jpg", ".jpeg"]
DEFAULT_HASH_ALGORITHMS = ["phash", "dhash"]


@dataclass
class Config:
    """Application settings. All fields mirror the documented defaults."""

    photo_root: str = DEFAULT_PHOTO_ROOT
    trash_root: str = DEFAULT_TRASH_ROOT
    cache_path: str = DEFAULT_CACHE_PATH
    file_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_FILE_EXTENSIONS))
    hash_algorithms: list[str] = field(default_factory=lambda: list(DEFAULT_HASH_ALGORITHMS))
    phash_threshold: int = 8
    dhash_threshold: int = 10
    ai_refinement: bool = False
    ai_model: str = "openai/clip-vit-base-patch32"
    ai_similarity_threshold: float = 0.85
    blur_enabled: bool = True
    blur_threshold_percentile: float = 10.0
    blur_min_absolute: float = 100.0

    def resolved_trash_root(self) -> Path:
        return Path(os.path.expanduser(self.trash_root))

    def resolved_cache_path(self) -> Path:
        return Path(os.path.expanduser(self.cache_path))


def config_dir() -> Path:
    return Path(os.path.expanduser(CONFIG_DIR))


def config_path() -> Path:
    return config_dir() / CONFIG_FILE


def _defaults() -> dict:
    return asdict(Config())


def _coerce(value, default):
    """Return *value* if it matches the type of *default*, else *default*."""
    if default is None:
        return value
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return type(default)(value)
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    if isinstance(default, list):
        if not isinstance(value, list):
            return default
        if not default:
            return value
        return [item for item in value if isinstance(item, type(default[0]))] or default
    return value


def load_config(path: Path | None = None) -> Config:
    """Load settings from *path* (default ``~/.config/similarity-tool/config.json``).

    A missing file yields the defaults. Invalid JSON, or values of the wrong
    type, fall back to defaults for the affected fields.
    """
    path = path or config_path()
    values = _defaults()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Could not read %s (%s); using default settings.", path, exc)
            raw = {}
        if not isinstance(raw, dict):
            log.error("%s does not contain a JSON object; using default settings.", path)
            raw = {}
        for key in fields(Config):
            if key.name in raw:
                values[key.name] = _coerce(raw[key.name], values[key.name])
    return Config(**values)


def ensure_config_file(path: Path | None = None) -> Path:
    """Create the config directory and a default ``config.json`` if missing.

    An existing file is never overwritten.
    """
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_defaults(), handle, indent=2)
            handle.write("\n")
    return path


def create_cache_dir(cfg: Config) -> Path:
    """Ensure the parent directory of the SQLite hash cache exists."""
    cache = cfg.resolved_cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    return cache


def copy_defaults() -> dict:
    """Return a fresh copy of the default settings (for tests and logging)."""
    return copy.deepcopy(_defaults())
