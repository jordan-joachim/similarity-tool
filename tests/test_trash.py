"""Tests for the trash subsystem: dated folders, UUID subfolders, relative
path preservation, JSON logging, same-day appends, and partial failures.

All tests use temporary directories; the real archive is never touched.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from similarity_tool.models import QueueItem, PhotoFile
from similarity_tool.trash import (
    dated_trash_dir,
    load_log,
    log_path_for,
    move_to_trash,
    write_log,
)


def _make_photo(root: Path, rel: str, content: bytes = b"jpeg-bytes") -> PhotoFile:
    """Create a real file under *root* and a PhotoFile pointing at it."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return PhotoFile(path=path, relative_path=rel, size=len(content), mtime=0.0)


def _queue_item(photo: PhotoFile, mode: str = "similarity") -> QueueItem:
    return QueueItem(photo=photo, mode=mode)


class TestDatedFolder:
    def test_dated_folder_uses_yyyymmdd(self, tmp_path):
        ts = datetime(2024, 5, 17, 10, 30, 0)
        assert dated_trash_dir(tmp_path, ts) == tmp_path / "2024-05-17"

    def test_different_days_create_different_folders(self, tmp_path):
        a = dated_trash_dir(tmp_path, datetime(2024, 5, 17))
        b = dated_trash_dir(tmp_path, datetime(2024, 5, 18))
        assert a != b


class TestMoveToTrash:
    def test_moves_file_and_preserves_relative_path(self, tmp_path):
        root = tmp_path / "Bilder"
        photo = _make_photo(root, "2024/05/IMG_1.jpg")
        result = move_to_trash(
            [_queue_item(photo)],
            tmp_path / "trash",
            timestamp=datetime(2024, 5, 17, 9, 0, 0),
        )
        assert len(result.moved) == 1
        assert not result.failures
        assert not photo.path.exists()
        trash_path = Path(result.moved[0].trash_path)
        assert trash_path.exists()
        assert trash_path.read_bytes() == b"jpeg-bytes"
        # Structure: trash/2024-05-17/<uuid>/2024/05/IMG_1.jpg
        parts = trash_path.relative_to(tmp_path / "trash" / "2024-05-17").parts
        assert parts[0] != "2024"  # the UUID subfolder
        assert parts[1:] == ("2024", "05", "IMG_1.jpg")

    def test_creates_trash_directories_automatically(self, tmp_path):
        root = tmp_path / "Bilder"
        photo = _make_photo(root, "2024/05/a.jpg")
        trash_root = tmp_path / "does" / "not" / "exist" / "trash"
        result = move_to_trash(
            [_queue_item(photo)],
            trash_root,
            timestamp=datetime(2024, 5, 17),
        )
        assert not result.failures
        assert Path(result.moved[0].trash_path).exists()

    def test_same_day_batches_use_distinct_uuid_subfolders(self, tmp_path):
        root = tmp_path / "Bilder"
        photo_a = _make_photo(root, "2024/05/a.jpg", content=b"aaa")
        photo_b = _make_photo(root, "2024/05/b.jpg", content=b"bbb")
        trash_root = tmp_path / "trash"
        ts = datetime(2024, 5, 17)
        first = move_to_trash([_queue_item(photo_a)], trash_root, timestamp=ts)
        second = move_to_trash([_queue_item(photo_b)], trash_root, timestamp=ts)
        assert not first.failures and not second.failures
        path_a = Path(first.moved[0].trash_path)
        path_b = Path(second.moved[0].trash_path)
        assert path_a.parent != path_b.parent  # distinct UUID subfolders
        assert path_a.exists() and path_b.exists()

    def test_same_relative_path_does_not_overwrite(self, tmp_path):
        root = tmp_path / "Bilder"
        photo_a = _make_photo(root, "2024/05/same.jpg", content=b"first")
        trash_root = tmp_path / "trash"
        ts = datetime(2024, 5, 17)
        first = move_to_trash([_queue_item(photo_a)], trash_root, timestamp=ts)
        assert not first.failures
        # A different file with the same relative path (e.g. re-imported after
        # the first was trashed) must land in a distinct UUID subfolder.
        photo_b = _make_photo(root, "2024/05/same.jpg", content=b"second")
        second = move_to_trash([_queue_item(photo_b)], trash_root, timestamp=ts)
        assert not second.failures
        path_a = Path(first.moved[0].trash_path)
        path_b = Path(second.moved[0].trash_path)
        assert path_a != path_b
        assert path_a.read_bytes() == b"first"
        assert path_b.read_bytes() == b"second"

    def test_missing_source_is_reported_and_not_logged(self, tmp_path):
        root = tmp_path / "Bilder"
        photo = _make_photo(root, "2024/05/gone.jpg")
        photo.path.unlink()  # simulate a file removed outside the app
        trash_root = tmp_path / "trash"
        result = move_to_trash(
            [_queue_item(photo)],
            trash_root,
            timestamp=datetime(2024, 5, 17),
        )
        assert not result.moved
        assert len(result.failures) == 1
        assert result.failures[0].item.photo.path == photo.path
        # No file was moved: the dated folder may exist (the UUID subfolder is
        # created before the move attempt) but must not contain the file.
        dated = trash_root / "2024-05-17"
        if dated.exists():
            for path in dated.rglob("*"):
                assert not path.is_file() or path.name != "gone.jpg"

    def test_partial_failure_keeps_successful_moves(self, tmp_path):
        root = tmp_path / "Bilder"
        good = _make_photo(root, "2024/05/good.jpg", content=b"good")
        bad = _make_photo(root, "2024/05/bad.jpg", content=b"bad")
        bad.path.unlink()  # this one will fail
        trash_root = tmp_path / "trash"
        result = move_to_trash(
            [_queue_item(good), _queue_item(bad)],
            trash_root,
            timestamp=datetime(2024, 5, 17),
        )
        assert len(result.moved) == 1
        assert len(result.failures) == 1
        assert Path(result.moved[0].trash_path).exists()
        assert not good.path.exists()
        assert result.failures[0].item.photo.path == bad.path


class TestTrashLog:
    def test_log_written_next_to_dated_folder(self, tmp_path):
        root = tmp_path / "Bilder"
        photo = _make_photo(root, "2024/05/a.jpg")
        trash_root = tmp_path / "trash"
        ts = datetime(2024, 5, 17, 8, 0, 0)
        result = move_to_trash([_queue_item(photo, mode="similarity")], trash_root, timestamp=ts)
        dated = trash_root / "2024-05-17"
        assert log_path_for(dated).exists()
        data = json.loads(log_path_for(dated).read_text())
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["original_path"] == str(photo.path)
        assert entry["trash_path"] == result.moved[0].trash_path
        assert entry["mode"] == "similarity"
        assert entry["timestamp"] == "2024-05-17T08:00:00"
        assert entry["file_size"] == len(b"jpeg-bytes")

    def test_log_records_per_item_mode_in_mixed_batch(self, tmp_path):
        root = tmp_path / "Bilder"
        sim = _make_photo(root, "2024/05/sim.jpg")
        blur = _make_photo(root, "2024/05/blur.jpg")
        trash_root = tmp_path / "trash"
        move_to_trash(
            [_queue_item(sim, mode="similarity"), _queue_item(blur, mode="blur")],
            trash_root,
            timestamp=datetime(2024, 5, 17),
        )
        data = json.loads(log_path_for(trash_root / "2024-05-17").read_text())
        modes = {e["mode"] for e in data["entries"]}
        assert modes == {"similarity", "blur"}

    def test_same_day_execution_appends_to_log(self, tmp_path):
        root = tmp_path / "Bilder"
        photo_a = _make_photo(root, "2024/05/a.jpg")
        photo_b = _make_photo(root, "2024/05/b.jpg")
        trash_root = tmp_path / "trash"
        ts = datetime(2024, 5, 17)
        move_to_trash([_queue_item(photo_a)], trash_root, timestamp=ts)
        move_to_trash([_queue_item(photo_b)], trash_root, timestamp=ts)
        data = json.loads(log_path_for(trash_root / "2024-05-17").read_text())
        assert len(data["entries"]) == 2

    def test_log_is_valid_json_and_parses(self, tmp_path):
        root = tmp_path / "Bilder"
        photo = _make_photo(root, "2024/05/a.jpg")
        trash_root = tmp_path / "trash"
        move_to_trash([_queue_item(photo)], trash_root, timestamp=datetime(2024, 5, 17))
        loaded = load_log(trash_root / "2024-05-17")
        assert len(loaded.entries) == 1
        assert loaded.entries[0].original_path == str(photo.path)

    def test_write_log_round_trips(self, tmp_path):
        dated = tmp_path / "2024-05-17"
        dated.mkdir()
        log_ = load_log(dated)
        log_.add(
            original_path=Path("/a/b.jpg"),
            trash_path=Path("/trash/2024-05-17/u/2024/05/b.jpg"),
            mode="blur",
            timestamp=datetime(2024, 5, 17, 12, 0, 0),
            file_size=42,
        )
        write_log(dated, log_)
        reloaded = load_log(dated)
        assert len(reloaded.entries) == 1
        assert reloaded.entries[0].mode == "blur"
        assert reloaded.entries[0].file_size == 42

    def test_corrupt_log_falls_back_to_empty(self, tmp_path):
        dated = tmp_path / "2024-05-17"
        dated.mkdir()
        log_path_for(dated).write_text("{not valid json")
        assert load_log(dated).entries == []
