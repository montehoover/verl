import os
import json
import configparser
from typing import Optional, Dict, Any, List


def read_local_config(config_filename: str, trusted_files: List[str]) -> Optional[Dict[str, Any]]:
    """
    Read configuration data from a local file and return it as a dictionary.

    Parameters:
    - config_filename: Name/path of the configuration file to read.
    - trusted_files: A list of strings representing trusted filenames/paths. If
      config_filename is not exactly present in this list, None is returned.

    Behavior:
    - If the file is not in the trusted_files list, returns None.
    - If the file does not exist, returns None.
    - Parses based on file extension:
        .json -> JSON
        .ini/.cfg -> INI using configparser
        .toml -> TOML using tomllib (Python 3.11+) or tomli if available
    - If the extension is unknown, attempts JSON, then INI, then TOML (if available).

    Raises:
    - TypeError for invalid argument types.
    - ValueError/RuntimeError for parse errors or unsupported formats.
    """
    if not isinstance(config_filename, str):
        raise TypeError("config_filename must be a string")
    if not isinstance(trusted_files, list) or not all(isinstance(x, str) for x in trusted_files):
        raise TypeError("trusted_files must be a list of strings")

    # Only proceed if the file is explicitly in the trusted list
    if config_filename not in trusted_files:
        return None

    if not os.path.exists(config_filename):
        return None

    ext = os.path.splitext(config_filename)[1].lower()

    def _parse_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON configuration must be an object at the top level")
        return data

    def _parse_ini(path: str) -> Dict[str, Any]:
        parser = configparser.ConfigParser(interpolation=None)
        with open(path, "r", encoding="utf-8") as f:
            parser.read_file(f)
        result: Dict[str, Any] = {}
        defaults = dict(parser.defaults())
        if defaults:
            result["DEFAULT"] = defaults
        for section in parser.sections():
            # raw=True to avoid interpolation
            result[section] = {k: v for k, v in parser.items(section, raw=True)}
        return result

    def _parse_toml(path: str) -> Dict[str, Any]:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore
            except ModuleNotFoundError:
                raise RuntimeError(
                    "TOML parsing requires Python 3.11+ (tomllib) or the 'tomli' package"
                )
        with open(path, "rb") as f:
            data = tomllib.load(f)
        if not isinstance(data, dict):
            raise ValueError("TOML configuration must parse to a dictionary")
        return data

    if ext == ".json":
        return _parse_json(config_filename)
    elif ext in (".ini", ".cfg"):
        return _parse_ini(config_filename)
    elif ext == ".toml":
        return _parse_toml(config_filename)
    else:
        # Try common formats in order
        for parser in (_parse_json, _parse_ini):
            try:
                return parser(config_filename)
            except Exception:
                pass
        try:
            return _parse_toml(config_filename)
        except Exception:
            pass
        raise ValueError(f"Unsupported or unrecognized configuration format for file: {config_filename}")
