import os
import json
import configparser
from typing import Dict, Any, List


def get_local_config_data(config_filename: str, approved_files: List[str]) -> Dict[str, Any]:
    """
    Read configuration data from a local file and return it as a dictionary.

    Parameters:
    - config_filename: The name/path of the configuration file to read.
    - approved_files: A list of approved filenames/paths. If config_filename is not
      present in this list, an empty dict is returned.

    Supports formats based on file extension:
    - .json -> JSON
    - .ini, .cfg, .conf -> INI (returned as nested dict with sections)
    - .toml -> TOML (requires Python 3.11+ or 'tomli' installed)
    - .yaml, .yml -> YAML (requires 'PyYAML' installed)

    If the file does not exist (or is not a regular file), returns an empty dict.
    """
    if not isinstance(config_filename, str):
        raise TypeError("config_filename must be a string")

    if not isinstance(approved_files, list) or not all(isinstance(x, str) for x in approved_files):
        raise TypeError("approved_files must be a list of strings")

    # Enforce approved file list
    if config_filename not in approved_files:
        return {}

    if not os.path.isfile(config_filename):
        return {}

    ext = os.path.splitext(config_filename)[1].lower()

    try:
        if ext == ".json":
            with open(config_filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}

        if ext in (".ini", ".cfg", ".conf"):
            parser = configparser.ConfigParser()
            parser.read(config_filename, encoding="utf-8")
            data: Dict[str, Any] = {}
            # Include DEFAULT section if present
            if parser.defaults():
                data["DEFAULT"] = dict(parser.defaults())
            # Include each section
            for section in parser.sections():
                # raw=True to avoid interpolation side effects
                data[section] = {k: v for k, v in parser.items(section, raw=True)}
            return data

        if ext == ".toml":
            try:
                import tomllib  # Python 3.11+
                with open(config_filename, "rb") as f:
                    data = tomllib.load(f)
                return data if isinstance(data, dict) else {}
            except (ImportError, ModuleNotFoundError):
                # Fallback to 'tomli' if available (Python <3.11)
                try:
                    import tomli  # type: ignore
                    with open(config_filename, "rb") as f:
                        data = tomli.load(f)
                    return data if isinstance(data, dict) else {}
                except Exception as e:
                    raise ValueError(
                        "TOML parsing requires Python 3.11+ (tomllib) or the 'tomli' package."
                    ) from e

        if ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise ValueError("YAML parsing requires the 'PyYAML' package.") from e
            with open(config_filename, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}

        # Unknown extension: try JSON as a sensible default
        with open(config_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        # File disappeared between isfile check and open
        return {}
