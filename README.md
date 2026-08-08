# Similarity Tool

A local GTK4 desktop application (Python/PyGObject) for finding and reviewing
**similar pictures** and **blurry / shaky pictures** in a chronologically
organised photo archive, then moving the unwanted ones to a local trash folder
instead of deleting them permanently.

## Features

- **Similarity mode**: finds clusters of visually similar JPEGs (bursts, HDR
  brackets, near-duplicates) using perceptual hashing (pHash + dHash), with an
  optional CPU-only CLIP/SigLIP second-stage refinement.
- **Blur mode**: scores every image with a Laplacian-variance sharpness metric
  and shows the worst offenders.
- Both modes use the same review workflow: a 2×4 thumbnail grid with
  per-image selection, a shared deletion queue, and a dated trash folder with a
  JSON log.

## Installation

Requirements:

- Python 3.11+ (developed on 3.14)
- GTK4 and PyGObject (system packages, e.g. `python3-gi` and `gir1.2-gtk-4.0`
  on Debian/Ubuntu)
- Pip packages are installed into a project virtual environment:

```bash
cd similarity-tool
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
```

The virtual environment uses `--system-site-packages` so the system-installed
GTK4/PyGObject bindings are reused. Detection dependencies (`Pillow`,
`imagehash`, `opencv-python`) are installed by pip. Optional AI refinement
(`transformers` + `torch`, CPU-only) can be installed with:

```bash
.venv/bin/pip install -e ".[ai]"
```

### Launching

```bash
cd similarity-tool
.venv/bin/python -m similarity_tool
```

or, once installed:

```bash
similarity-tool
```

A desktop entry can point to either command.

## Configuration

On first launch the application creates its configuration directory and a
default `config.json`:

```
~/.config/similarity-tool/config.json
```

| Setting | Default | Purpose |
| --- | --- | --- |
| `photo_root` | `/run/media/joachim/LinStorage/Media/Sammlung/Bilder` | Photo archive root; scanned as `YYYY/MM` folders |
| `trash_root` | `~/.local/share/similarity-tool/trash` | Folder that executed deletions are moved to |
| `cache_path` | `~/.cache/similarity-tool/hashes.sqlite3` | SQLite cache for hashes and blur scores |
| `file_extensions` | `[".jpg", ".jpeg"]` | File types the scanner considers |
| `hash_algorithms` | `["phash", "dhash"]` | Perceptual hashes computed for each image |
| `phash_threshold` | `8` | Maximum pHash Hamming distance for a match |
| `dhash_threshold` | `10` | Maximum dHash Hamming distance for a match |
| `ai_refinement` | `false` | Enable the optional CLIP/SigLIP second stage |
| `ai_model` | `openai/clip-vit-base-patch32` | Model used by the AI stage |
| `ai_similarity_threshold` | `0.85` | Minimum cosine similarity kept by the AI stage |
| `blur_enabled` | `true` | Enable Blur mode |
| `blur_threshold_percentile` | `10` | Bottom N% of a month's images flagged as blurry |
| `blur_min_absolute` | `100.0` | Absolute sharpness floor; anything below is flagged |

Other runtime directories, created automatically:

- Cache: `~/.cache/similarity-tool/`
- Trash: `~/.local/share/similarity-tool/trash/`

If the config file is missing, malformed, or contains values of the wrong
type, the application logs an error and uses the defaults above.

## Usage

1. Choose a mode (Similarity or Blur) in the toolbar.
2. Pick a year and month from the left-hand tree.
3. Press **Scan**. Results appear in the left pane as clusters (Similarity) or
   ranked candidates (Blur); each result shows up to 8 images in the 2×4 grid.
4. Check the thumbnails you want to remove and press **Add to Queue** (or `Q`).
5. Review the staged files in the **Queue** tab, then press **Execute Queue**
   (or `E`) and confirm. Files are moved to the trash folder, never deleted.

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Space` | Toggle selection of the focused grid cell |
| `A` | Select all cells in the current grid |
| `N` | Select none |
| `Q` | Queue the selected images |
| `E` | Execute the queue |
| `D` | Discard the queue |
| `Esc` | Quit |

### Restoring files from trash

Executed deletions are moves, not permanent deletions. Files land in

```
~/.local/share/similarity-tool/trash/YYYY-MM-DD/<uuid>/<relative-path>
```

where `<relative-path>` mirrors the file's position under `photo_root`. To
restore a file, move it back to its original path, for example:

```bash
cp "~/.local/share/similarity-tool/trash/2024-05-17/<uuid>/2024/05/IMG_1234.jpg" \
   "/run/media/joachim/LinStorage/Media/Sammlung/Bilder/2024/05/IMG_1234.jpg"
```

Each execution also writes a JSON log (`trash.log.json`) next to the dated
folder recording every file's original path, trash path, source mode,
timestamp, and size.

## Development

```bash
.venv/bin/python -m pytest        # run the test suite
.venv/bin/python -m ruff check src  # lint (optional)
```

## License

MIT.
