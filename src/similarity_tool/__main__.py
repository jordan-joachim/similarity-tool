"""Command-line entry point for the Similarity Tool.

Supports ``python -m similarity_tool`` and the ``similarity-tool`` console script.
Imports are kept at call time so that ``--help`` works even when optional GUI
or detection dependencies are missing.
"""

import argparse
import sys


def _check_dependencies() -> None:
    """Verify that required runtime dependencies are importable.

    Raises a SystemExit with a clear, user-facing message (and a non-zero exit
    code) instead of letting a confusing import traceback escape.
    """
    missing = []
    for name in ("gi", "PIL", "imagehash", "cv2"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)

    if "gi" in missing:
        sys.stderr.write(
            "Error: PyGObject (Python bindings for GTK) is not installed.\n"
            "Install it with your system package manager, e.g.\n"
            "  sudo apt install python3-gi gir1.2-gtk-4.0\n"
            "then recreate the virtual environment with --system-site-packages.\n"
        )
    if "PIL" in missing:
        sys.stderr.write("Error: Pillow is not installed. Run: pip install -e .\n")
    if "imagehash" in missing:
        sys.stderr.write("Error: imagehash is not installed. Run: pip install -e .\n")
    if "cv2" in missing:
        sys.stderr.write(
            "Error: opencv-python is not installed. Run: pip install -e .\n"
        )
    if missing:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="similarity-tool",
        description="Find and review similar and blurry photos.",
    )
    parser.add_argument(
        "--version", action="version", version=f"similarity-tool {__import__('similarity_tool').__version__}"
    )
    parser.parse_args(argv)

    _check_dependencies()

    try:
        import gi  # noqa: PLC0415

        gi.require_version("Gtk", "4.0")
        from gi.repository import Gtk  # noqa: PLC0415
    except (ImportError, ValueError) as exc:
        sys.stderr.write(f"Error: GTK4 (PyGObject) is not usable: {exc}\n")
        sys.stderr.write(
            "Install it with your system package manager, e.g.\n"
            "  sudo apt install python3-gi gir1.2-gtk-4.0\n"
        )
        return 1

    from similarity_tool import gui  # noqa: PLC0415

    app = gui.SimilarityToolApplication()
    return app.run(None)


if __name__ == "__main__":
    raise SystemExit(main())
