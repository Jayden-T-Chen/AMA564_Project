"""Single entry point for the frozen configuration files."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def loadConfig(name):
    """Return configs/<name>.yaml as a dict."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def getPath(config, dottedPath):
    """Read a two-level key: "entropyGate.keepFraction" -> config[...][...]."""
    section, key = dottedPath.split(".")
    return config[section][key]


def setPath(config, dottedPath, value):
    """Write a two-level key in place and return the config."""
    section, key = dottedPath.split(".")
    config[section][key] = value
    return config
