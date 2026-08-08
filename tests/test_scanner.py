"""Tests for the file scanner: discovery, extension filtering, EXIF, metadata."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from similarity_tool.models import PhotoFile
from similarity_tool.scanner import (
    list_year_months,
    read_date_taken,
    scan_month,
)


def _make_jpeg(path: Path, exif: dict[int, str] | None = None) -> None:
    """Write a tiny valid JPEG, optionally with EXIF tags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (4, 4), color=(10, 20, 30))
    if exif:
        exif_obj = Image.Exif()
        for tag, value in exif.items():
            exif_obj[tag] = value
        img.save(path, format="JPEG", exif=exif_obj)
    else:
        img.save(path, format="JPEG")


def _make_photo_root(tmp_path: Path) -> Path:
    """Create a photo root with a 2024/05 month and non-numeric siblings."""
    root = tmp_path / "Bilder"
    _make_jpeg(root / "2024" / "05" / "a.jpg")
    _make_jpeg(root / "2024" / "05" / "b.JPEG")
    _make_jpeg(root / "2024" / "05" / "c.Jpg")
    _make_jpeg(root / "2024" / "05" / "d.png")
    (root / "2024" / "05" / "e.txt").write_text("not an image")
    (root / "2024" / "05" / "f").write_text("no extension")
    _make_jpeg(root / "2024" / "05" / "sub" / "g.jpeg")
    _make_jpeg(root / "2024" / "05" / "sub" / "deep" / "h.jpg")
    # Non-numeric top-level folder must be ignored.
    _make_jpeg(root / "Christiane" / "2024" / "05" / "x.jpg")
    # Non-numeric month folder must be ignored.
    _make_jpeg(root / "2024" / "misc" / "y.jpg")
    return root


class TestScanMonthDiscovery:
    def test_finds_only_configured_extensions_case_insensitive(self, tmp_path):
        root = _make_photo_root(tmp_path)
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        names = sorted(p.path.name for p in photos)
        assert names == ["a.jpg", "b.JPEG", "c.Jpg", "g.jpeg", "h.jpg"]
        assert "d.png" not in names
        assert "e.txt" not in names
        assert "f" not in names

    def test_handles_subdirectories(self, tmp_path):
        root = _make_photo_root(tmp_path)
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        rel = sorted(p.relative_path for p in photos)
        assert "2024/05/sub/g.jpeg" in rel
        assert "2024/05/sub/deep/h.jpg" in rel

    def test_ignores_non_numeric_top_level_folders(self, tmp_path):
        root = _make_photo_root(tmp_path)
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        rel = [p.relative_path for p in photos]
        assert not any(p.startswith("Christiane") for p in rel)
        assert not any(p.startswith("2024/misc") for p in rel)

    def test_missing_month_returns_empty_list(self, tmp_path):
        root = _make_photo_root(tmp_path)
        assert scan_month(root, "1999", "12", [".jpg", ".jpeg"]) == []
        assert scan_month(root, "2024", "99", [".jpg", ".jpeg"]) == []

    def test_custom_extensions_are_honored(self, tmp_path):
        root = _make_photo_root(tmp_path)
        photos = scan_month(root, "2024", "05", [".png"])
        assert [p.path.name for p in photos] == ["d.png"]

    def test_empty_extension_list_finds_nothing(self, tmp_path):
        root = _make_photo_root(tmp_path)
        assert scan_month(root, "2024", "05", []) == []

    def test_extension_without_dot_is_normalized(self, tmp_path):
        root = _make_photo_root(tmp_path)
        photos = scan_month(root, "2024", "05", ["jpg"])
        assert "a.jpg" in [p.path.name for p in photos]


class TestScanMonthMetadata:
    def test_returns_absolute_and_relative_paths(self, tmp_path):
        root = _make_photo_root(tmp_path)
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        a = next(p for p in photos if p.path.name == "a.jpg")
        assert a.path.is_absolute()
        assert a.path == root / "2024" / "05" / "a.jpg"
        assert a.relative_path == "2024/05/a.jpg"

    def test_returns_size_and_mtime(self, tmp_path):
        root = _make_photo_root(tmp_path)
        target = root / "2024" / "05" / "a.jpg"
        _make_jpeg(target)
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        a = next(p for p in photos if p.path.name == "a.jpg")
        stat = target.stat()
        assert a.size == stat.st_size
        assert a.size > 0
        assert a.mtime == pytest.approx(stat.st_mtime)

    def test_results_are_sorted_deterministically(self, tmp_path):
        root = _make_photo_root(tmp_path)
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        rel = [p.relative_path for p in photos]
        assert rel == sorted(rel)

    def test_returns_photofile_instances(self, tmp_path):
        root = _make_photo_root(tmp_path)
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        assert photos
        assert all(isinstance(p, PhotoFile) for p in photos)


class TestExif:
    def test_datetime_original_is_read(self, tmp_path):
        root = tmp_path / "Bilder"
        _make_jpeg(
            root / "2024" / "05" / "a.jpg",
            exif={36867: "2024:05:17 12:34:56", 306: "2023:01:02 03:04:05"},
        )
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        assert photos[0].date_taken == "2024-05-17T12:34:56"

    def test_falls_back_to_datetime_tag(self, tmp_path):
        root = tmp_path / "Bilder"
        _make_jpeg(root / "2024" / "05" / "a.jpg", exif={306: "2023:01:02 03:04:05"})
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        assert photos[0].date_taken == "2023-01-02T03:04:05"

    def test_falls_back_to_mtime_when_no_exif(self, tmp_path):
        root = tmp_path / "Bilder"
        target = root / "2024" / "05" / "a.jpg"
        _make_jpeg(target)
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        expected = datetime.fromtimestamp(target.stat().st_mtime).isoformat(timespec="seconds")  # noqa: DTZ006
        assert photos[0].date_taken == expected

    def test_corrupt_image_does_not_crash_and_falls_back(self, tmp_path):
        root = tmp_path / "Bilder"
        target = root / "2024" / "05" / "broken.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\xff\xd8\xff\xe0 not a real jpeg")
        _make_jpeg(root / "2024" / "05" / "ok.jpg")
        photos = scan_month(root, "2024", "05", [".jpg", ".jpeg"])
        names = sorted(p.path.name for p in photos)
        assert names == ["broken.jpg", "ok.jpg"]
        broken = next(p for p in photos if p.path.name == "broken.jpg")
        expected = datetime.fromtimestamp(target.stat().st_mtime).isoformat(timespec="seconds")  # noqa: DTZ006
        assert broken.date_taken == expected

    def test_read_date_taken_returns_none_for_missing_file(self, tmp_path):
        assert read_date_taken(tmp_path / "nope.jpg") is None


class TestListYearMonths:
    def test_returns_only_numeric_year_month_pairs(self, tmp_path):
        root = tmp_path / "Bilder"
        for folder in ("2004/01", "2004/02", "2005/03", "Christiane/2024/05", "2024/misc"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        pairs = list_year_months(root)
        assert pairs == [("2004", "01"), ("2004", "02"), ("2005", "03")]

    def test_ignores_non_four_digit_years(self, tmp_path):
        root = tmp_path / "Bilder"
        for folder in ("24/01", "2024a/01", "2024/1"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        assert list_year_months(root) == []

    def test_missing_root_returns_empty(self, tmp_path):
        assert list_year_months(tmp_path / "nope") == []
