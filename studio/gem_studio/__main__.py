"""Launch GEM-X Studio in a native window (OS webview) or the default browser.

    python -m gem_studio              # native window via pywebview (WebView2 on Windows)
    python -m gem_studio --browser    # serve and open the default browser instead
"""

import argparse
import atexit
import os
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .paths import JOBS_DIR, MEDIA_FILE_TYPES, PACKAGE_DIR, PORTABLE, TOOLS_DIR
from .power import opt_out_of_power_throttling


def _free_port(preferred):
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return probe.getsockname()[1]
    raise RuntimeError("No free local port")


class NativeApi:
    """Methods exposed to the page as ``window.pywebview.api.*``."""

    def __init__(self):
        self._window = None

    def _dialog(self, kind):
        import webview

        enum = getattr(webview, "FileDialog", None)
        if enum is not None:
            return getattr(enum, kind)
        return getattr(webview, f"{kind}_DIALOG")

    def pick_media(self):
        from . import inputs

        paths = self._window.create_file_dialog(
            self._dialog("OPEN"), allow_multiple=True, file_types=MEDIA_FILE_TYPES
        )
        records, errors = [], []
        for path in paths or []:
            try:
                records.append(inputs.register(path))
            except Exception as error:
                errors.append({"path": path, "error": str(error)})
        return {"inputs": records, "errors": errors}

    def save_file(self, job_id, relative):
        directory = (JOBS_DIR / job_id).resolve()
        source = (directory / relative).resolve()
        if directory not in source.parents or not source.is_file():
            return None
        target = self._window.create_file_dialog(self._dialog("SAVE"), save_filename=source.name)
        if not target:
            return None
        target = target if isinstance(target, str) else target[0]
        shutil.copyfile(source, target)
        return target

    def open_external(self, url):
        if url.startswith(("https://", "http://")):
            webbrowser.open(url)


def _windowless_logging():
    """pythonw has no console: keep server/worker diagnostics in a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from .paths import DATA_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stream = open(DATA_DIR / "studio.log", "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def _windows_identity():
    """Own taskbar group instead of pythonw's (the window icon is set by pywebview)."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GEMX.Studio")
    except (AttributeError, OSError):
        pass


def _window_icon():
    """pywebview's Windows backend loads the icon with System.Drawing.Icon, which only
    accepts .ico files (a PNG crashes the process); other platforms take the PNG."""
    icon = PACKAGE_DIR / ("icon.ico" if sys.platform == "win32" else "icon.png")
    return str(icon) if icon.is_file() else None


def main(argv=None):
    _windowless_logging()
    opt_out_of_power_throttling()
    parser = argparse.ArgumentParser(description="GEM-X Studio")
    parser.add_argument("--browser", action="store_true", help="Use the default web browser")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Serve only (browser mode)")
    parser.add_argument("--dev", action="store_true", help="Disable token checks for Vite dev")
    parser.add_argument("--debug", action="store_true", help="Enable webview devtools")
    args = parser.parse_args(argv)

    native = not args.browser and not args.no_open
    if native:
        try:
            import webview  # noqa: F401
        except ImportError:
            print("pywebview is not installed; opening the browser instead.", flush=True)
            native = False

    import uvicorn

    from .app import create_app

    port = args.port if args.dev else _free_port(args.port)
    app = create_app(port=port, native=native, dev=args.dev)
    atexit.register(lambda: [app.state.jobs.cancel(job["id"]) for job in app.state.jobs.list()
                             if job["status"] == "running"])  # fmt: skip
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    url = f"http://127.0.0.1:{port}/"

    if not native:
        print(f"GEM-X Studio running at {url}  (Ctrl+C to stop)", flush=True)
        if not args.no_open:
            threading.Timer(1.0, webbrowser.open, args=(url,)).start()
        server.run()
        return

    import webview

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    api = NativeApi()
    window = webview.create_window(
        "GEM-X Studio",
        url,
        js_api=api,
        width=1480,
        height=940,
        min_size=(1040, 700),
        background_color="#0a0a0b",
        text_select=False,
    )
    api._window = window
    _windows_identity()
    if PORTABLE:
        storage = TOOLS_DIR / "webview"
    else:
        storage = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "GEM-X Studio" / "webview"
    webview.start(
        debug=args.debug,
        icon=_window_icon(),
        private_mode=False,
        storage_path=str(storage),
    )
    server.should_exit = True
    thread.join(timeout=5)
    sys.exit(0)


if __name__ == "__main__":
    main()
