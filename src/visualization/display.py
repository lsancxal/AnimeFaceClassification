"""Open saved figures in a local viewer when available."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _should_open_figures(show: bool) -> bool:
    if not show:
        return False
    # Headless / Docker: no GUI display
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    if os.environ.get("SHOW_FIGURES", "1").lower() in {"0", "false", "no"}:
        return False
    return True


def show_saved_figure(save_path, show=True):
    """Open a saved figure in the default image viewer when possible."""
    if not _should_open_figures(show) or not save_path:
        return

    absolute_path = os.path.abspath(save_path)
    if not os.path.exists(absolute_path):
        return

    print(f"  Opening: {absolute_path}")
    try:
        if os.name == "nt":
            os.startfile(absolute_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", absolute_path], check=False)
        else:
            opener = shutil.which("xdg-open")
            if opener is None:
                return
            subprocess.run([opener, absolute_path], check=False)
    except (FileNotFoundError, OSError):
        # Container / headless environments may not have a viewer.
        return
