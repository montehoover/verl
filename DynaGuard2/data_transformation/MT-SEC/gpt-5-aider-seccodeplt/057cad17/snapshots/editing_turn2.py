from __future__ import annotations

import json
import os
from typing import Optional, Dict, Any

try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None  # type: ignore

import configparser


def _parse_json(filename: str) -> Dict[str, Any]:
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON configuration must be a JSON object")
    return data


def _parse_ini(filename: str) -> Dict[str, Any]:
    parser = configparser.ConfigParser()
    # Use read_file to raise if file can't be read; read() returns list of files read.
    # Using read() with encoding to support UTF-8.
    read_files = parser.read(filename, encoding="utf-8")
    if not read_files:
        raise ValueError("Failed to read INI configuration")
    result: Dict[str, Any] = {}
    # Include DEFAULT section if present
    defaults = parser.defaults()
    if defaults:
        result["DEFAULT"] = dict(defaults)
    # Include all explicit sections
    for section in parser.sections():
        result[section] = dict(parser.items(section))
    return result


def _parse_toml(filename: str) -> Dict[str, Any]:
    if tomllib is None:
        raise ImportError("TOML support requires Python 3.11+ (tomllib)")
    with open(filename, "rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError("TOML configuration must be a table (dict)")
    return data


def read_local_config(filename: str, approved_files: list[str]) -> Optional[Dict[str, Any]]:
    """
    Read configuration data from a local file, restricted to an approved list.

    Args:
        filename: Path to the configuration file.
        approved_files: List of approved filenames/paths that are allowed to be read.

    Returns:
        A dictionary with the configuration data. If the file does not exist or is not
        in the approved list, returns None.

    Supported formats:
        - .json
        - .ini, .cfg, .conf (INI format)
        - .toml (Python 3.11+)
        If the extension is unrecognized, attempts JSON, then INI, then TOML (if available).
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")

    if not isinstance(approved_files, list):
        raise TypeError("approved_files must be a list of strings")

    if filename not in approved_files:
        return None

    if not os.path.isfile(filename):
        return None

    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".json":
            return _parse_json(filename)
        if ext in (".ini", ".cfg", ".conf"):
            return _parse_ini(filename)
        if ext == ".toml":
            return _parse_toml(filename)

        # Unknown extension: try common formats in order
        try:
            return _parse_json(filename)
        except Exception:
            pass

        try:
            return _parse_ini(filename)
        except Exception:
            pass

        if tomllib is not None:
            try:
                return _parse_toml(filename)
            except Exception:
                pass

        raise ValueError("Unrecognized configuration format or failed to parse")
    except FileNotFoundError:
        # File might be removed between the isfile check and open/read
        return None
