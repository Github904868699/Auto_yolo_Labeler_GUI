"""Utilities for resolving resource paths in both source and frozen builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union


def get_base_path() -> Path:
    """Return the base directory that holds bundled resources."""
    if hasattr(sys, "_MEIPASS"):
        meipass = getattr(sys, "_MEIPASS")
        if meipass:
            return Path(meipass)
    return Path(__file__).resolve().parents[1]


def resource_path(relative_path: Union[str, os.PathLike[str]]) -> str:
    """Resolve *relative_path* against the application resource root.

    When the app is frozen by PyInstaller, resources are extracted beneath
    ``sys._MEIPASS``. During local development we fall back to the project
    root, allowing the same call sites to load icons, configuration files,
    or model checkpoints regardless of execution environment.
    """

    path = Path(relative_path)
    if path.is_absolute():
        return str(path)

    base = get_base_path()
    return str((base / path).resolve())
