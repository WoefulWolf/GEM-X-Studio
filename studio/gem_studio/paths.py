"""Filesystem locations shared by the server, the worker and the installer."""

import os
import shutil
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"


def _find_repo_root():
    configured = os.environ.get("GEM_REPO_ROOT")
    if configured:
        return Path(configured).resolve()
    for candidate in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (candidate / "setup.cfg").is_file() and (candidate / "gem").is_dir():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = _find_repo_root()
WEIGHTS_DIR = Path(os.environ.get("GEM_WEIGHTS_DIR", REPO_ROOT / "weights")).resolve()
SOMA_DIR = Path(os.environ.get("GEM_SOMA_ASSETS", REPO_ROOT / "assets" / "soma")).resolve()
DATA_DIR = Path(os.environ.get("GEM_STUDIO_DATA", REPO_ROOT / "studio_data")).resolve()
INPUTS_DIR = DATA_DIR / "inputs"
JOBS_DIR = DATA_DIR / "jobs"
SETTINGS_FILE = DATA_DIR / "settings.json"
MOTION_CONFIG = REPO_ROOT / "docs" / "configs" / "body_video_native77.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
MEDIA_FILE_TYPES = (
    "Images and videos ("
    + ";".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
    + ")",
    "All files (*.*)",
)


def ensure_dirs():
    for directory in (INPUTS_DIR, JOBS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def ffmpeg_executable():
    """System FFmpeg first, then the binary bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ensure_ffmpeg_on_path():
    """GEM calls a bare ``ffmpeg``; expose the bundled binary under that name.

    The imageio-ffmpeg binary has a versioned filename, so a copy named
    ``ffmpeg(.exe)`` is placed next to the Python interpreter's scripts.
    """
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    source = ffmpeg_executable()
    if source is None:
        return None
    bin_dir = DATA_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if not target.exists():
        shutil.copy2(source, target)
        target.chmod(0o755)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(target)


def media_kind(path):
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None
