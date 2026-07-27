import os


def show_saved_figure(save_path, show=True):
    """Open a saved figure in the default image viewer (avoids matplotlib/tkinter issues)."""
    if not show or not save_path:
        return

    absolute_path = os.path.abspath(save_path)
    if not os.path.exists(absolute_path):
        return

    print(f"  Opening: {absolute_path}")
    if os.name == 'nt':
        os.startfile(absolute_path)
    else:
        import subprocess
        import sys

        if sys.platform == 'darwin':
            subprocess.run(['open', absolute_path], check=False)
        else:
            subprocess.run(['xdg-open', absolute_path], check=False)
