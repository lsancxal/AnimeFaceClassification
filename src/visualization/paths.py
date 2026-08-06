import os
from datetime import datetime

_RUN_TIMESTAMP = None


def set_run_timestamp(timestamp=None):
    """Set one timestamp for the whole training run."""
    global _RUN_TIMESTAMP
    _RUN_TIMESTAMP = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return _RUN_TIMESTAMP


def get_run_timestamp():
    """Return the current run timestamp, creating one if needed."""
    if _RUN_TIMESTAMP is None:
        return set_run_timestamp()
    return _RUN_TIMESTAMP


def output_path(filename):
    """Build a path under outputs/ with a stable filename."""
    return os.path.join("outputs", filename)


def add_run_timestamp(fig):
    """Draw the run timestamp in the bottom-right corner of a figure."""
    fig.text(
        0.99,
        0.01,
        f"Run: {get_run_timestamp()}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
        transform=fig.transFigure,
    )
