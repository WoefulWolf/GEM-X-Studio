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

# Portable install (install.bat -Portable): caches that libraries would otherwise keep
# in the user profile live in .tools\cache instead, so the folder holds everything.
TOOLS_DIR = REPO_ROOT / ".tools"
PORTABLE = (TOOLS_DIR / "portable").is_file()
CACHE_DIR = TOOLS_DIR / "cache"
PORTABLE_CACHES = {
    "HF_HOME": "huggingface",  # Hugging Face hub metadata and the Xet transfer cache
    "WARP_CACHE_PATH": "warp",  # NVIDIA Warp kernels compiled by SOMA
    "CUDA_CACHE_PATH": "nvidia",  # CUDA JIT compute cache
    "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton",
    "MPLCONFIGDIR": "matplotlib",
    "XDG_CACHE_HOME": ".",
    "UV_CACHE_DIR": "uv",
    "npm_config_cache": "npm",
}
if PORTABLE:
    # Set before huggingface_hub, warp or torch are imported; subprocesses inherit them.
    # Variables the user set explicitly are left alone.
    for _name, _folder in PORTABLE_CACHES.items():
        os.environ.setdefault(_name, str(CACHE_DIR / _folder))
    # Model files go straight to weights/; the Xet chunk cache would only keep a second copy.
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

# Optional ViPE (moving-camera Body Motion) in its own environment: install.bat -Vipe.
VIPE_VENV = REPO_ROOT / ".venv-vipe"

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


def python_executable():
    """Interpreter for background processes.

    The app window runs under the windowless ``pythonw.exe``, which has no usable
    stdout; workers need the console ``python.exe`` next to it (they are still
    started hidden with CREATE_NO_WINDOW).
    """
    executable = Path(sys.executable)
    console = executable.with_name("python.exe")
    if executable.name.lower() == "pythonw.exe" and console.is_file():
        return str(console)
    return sys.executable


def bundled_vipe():
    """The ViPE executable set up by the installer, or None."""
    executable = VIPE_VENV / ("Scripts/vipe.exe" if sys.platform == "win32" else "bin/vipe")
    return str(executable) if executable.is_file() else None


def vipe_pipeline(executable):
    """ViPE pipeline for an install, or None for GEM's default (pose_only_long).

    pose_only_long's long-sequence pose graph needs gtsam, which has no Windows build;
    pose_only tracks the same camera without it. Unknown layouts keep the default.
    """
    environment = Path(executable).resolve().parent.parent
    if not (environment / "pyvenv.cfg").is_file():
        return None
    gtsam = [environment / "Lib" / "site-packages" / "gtsam"]
    gtsam += environment.glob("lib/python3*/site-packages/gtsam")
    return None if any(path.is_dir() for path in gtsam) else "pose_only"


def media_kind(path):
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None
