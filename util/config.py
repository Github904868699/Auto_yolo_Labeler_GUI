from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path.home() / ".auto_yolo_labeler"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> Dict[str, Any]:
    """Load persisted application configuration.

    Returns an empty dictionary when the configuration file does not exist or
    cannot be parsed.
    """
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        # Broken configuration files should not crash the application.
        return {}


def save_config(config: Dict[str, Any]) -> None:
    """Persist configuration values to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)


def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience helper to merge and persist configuration updates."""
    config = load_config()
    config.update({k: v for k, v in updates.items() if v is not None})
    save_config(config)
    return config
