"""Environment report, probed in a subprocess so the server never imports torch."""

import json
import platform
import subprocess
import sys
import threading

from .paths import (
    CACHE_DIR,
    DATA_DIR,
    PORTABLE,
    REPO_ROOT,
    SOMA_DIR,
    WEIGHTS_DIR,
    ffmpeg_executable,
    python_executable,
)

_PROBE = r"""
import json, importlib.util
info = {}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    info["gpus"] = [
        {"name": torch.cuda.get_device_name(i),
         "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
        for i in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else []
    mps = getattr(torch.backends, "mps", None)
    info["mps_available"] = bool(mps and mps.is_available())
except Exception as error:
    info["torch_error"] = f"{type(error).__name__}: {error}"
try:
    import gem  # noqa: F401
    info["gem"] = True
except Exception as error:
    info["gem"] = False
    info["gem_error"] = f"{type(error).__name__}: {error}"
try:
    import onnxruntime
    info["onnxruntime"] = onnxruntime.__version__
    info["onnx_providers"] = onnxruntime.get_available_providers()
except Exception:
    info["onnxruntime"] = None
info["open3d"] = importlib.util.find_spec("open3d") is not None
print(json.dumps(info))
"""

_cache = {}
_lock = threading.Lock()


def _probe():
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        output = subprocess.run(
            [python_executable(), "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=180,
            creationflags=flags,
            cwd=REPO_ROOT,
        )
        return json.loads(output.stdout.strip().splitlines()[-1])
    except Exception as error:
        return {"probe_error": f"{type(error).__name__}: {error}"}


def start_probe():
    def run():
        result = _probe()
        with _lock:
            _cache["runtime"] = result

    threading.Thread(target=run, daemon=True).start()


def info(native=False):
    with _lock:
        runtime = _cache.get("runtime")
    try:
        import webview  # noqa: F401

        has_webview = True
    except ImportError:
        has_webview = False
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "repo_root": str(REPO_ROOT),
        "data_dir": str(DATA_DIR),
        "portable": PORTABLE,
        "cache_dir": str(CACHE_DIR) if PORTABLE else None,
        "weights_dir": str(WEIGHTS_DIR),
        "soma_dir": str(SOMA_DIR),
        "ffmpeg": ffmpeg_executable(),
        "native": native,
        "pywebview": has_webview,
        "runtime": runtime,
    }
