"""File discovery and EXIF metadata extraction for the Similarity Tool.

The scanner is the only component that enumerates the photo archive. It walks
``photo_root/<YYYY>/<MM>/`` (including subdirectories), keeps only files whose
extension is in the configured list (case-insensitive), and returns
:class:`~similarity_tool.models.PhotoFile` records with size, mtime, absolute
and relative paths, and an EXIF capture time when one can be determined.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from similarity_tool.models import PhotoFile

log = logging.getLogger(__name__)

# EXIF tag IDs (Pillow's Image.Exif uses the raw tag numbers).
TAG_DATETIME_ORIGINAL = 36867  # 0x9003
TAG_DATETIME = 306  # 0x0132

# EXIF timestamps look like "2024:05:17 12:34:56".
_EXIF_FORMAT = "%Y:%m:%d %H:%M:%S"


def _normalize_extensions(extensions: list[str]) -> set[str]:
    """Lowercase the configured extensions and ensure they start with a dot."""
    normalized: set[str] = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def read_date_taken(path: Path) -> str | None:
    """Return an ISO capture time from EXIF for *path*, or ``None``.

    The EXIF ``DateTimeOriginal`` tag is preferred, then ``DateTime``.
    ``scan_month`` applies the final fallback to the file's modification time
    when this returns ``None``. A file that cannot be decoded by Pillow is not
    an error: the caller still gets a usable ``PhotoFile`` with the mtime
    fallback.
    """
    try:
        with Image.open(path) as image:
            exif = image.getexif()
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
        log.warning("Could not read EXIF from %s; using file mtime.", path)
        return None

    for tag in (TAG_DATETIME_ORIGINAL, TAG_DATETIME):
        raw = exif.get(tag)
        if not raw:
            continue
        try:
            # EXIF timestamps are naive local wall-clock times (no timezone).
            parsed = datetime.strptime(str(raw).strip(), _EXIF_FORMAT)  # noqa: DTZ007
        except ValueError:
            log.warning("Unrecognized EXIF timestamp %r in %s; using file mtime.", raw, path)
            continue
        return parsed.isoformat(timespec="seconds")
    return None


def scan_month(
    photo_root: str | Path,
    year: str,
    month: str,
    file_extensions: list[str],
) -> list[PhotoFile]:
    """Enumerate supported images under ``photo_root/<year>/<month>/``.

    Only files whose extension is in *file_extensions* (compared
    case-insensitively) are returned, including files in subdirectories of the
    month folder. Non-numeric sibling folders are never entered because the
    walk starts inside the resolved ``<year>/<month>`` path. Results are sorted
    by relative path for deterministic ordering.
    """
    root = Path(photo_root)
    month_dir = root / year / month
    extensions = _normalize_extensions(file_extensions)

    photos: list[PhotoFile] = []
    if not month_dir.is_dir():
        return photos

    for path in sorted(month_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            stat = path.stat()
        except OSError:
            log.warning("Could not stat %s; skipping.", path)
            continue
        date_taken = read_date_taken(path)
        if date_taken is None:
            # Fall back to the file's modification time when no usable EXIF
            # capture time exists (architecture: DateTimeOriginal -> DateTime
            # -> file mtime). Local naive time keeps the fallback consistent
            # with naive EXIF timestamps.
            date_taken = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")  # noqa: DTZ006
        photos.append(
            PhotoFile(
                path=path,
                relative_path=path.relative_to(root).as_posix(),
                size=stat.st_size,
                mtime=stat.st_mtime,
                date_taken=date_taken,
            )
        )
    return photos


def list_year_months(photo_root: str | Path) -> list[tuple[str, str]]:
    """Return sorted ``(year, month)`` pairs for numeric ``YYYY/MM`` folders.

    Top-level folders whose names are not exactly four decimal digits, and
    month folders that are not exactly two decimal digits, are ignored. This
    is what the GUI year/month tree is built from.
    """
    root = Path(photo_root)
    pairs: list[tuple[str, str]] = []
    if not root.is_dir():
        return pairs
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit() or len(year_dir.name) != 4:
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.isdigit() or len(month_dir.name) != 2:
                continue
            pairs.append((year_dir.name, month_dir.name))
    return pairs
