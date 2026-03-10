from copy import deepcopy
from typing import Any, Dict

# Example in-memory configuration "database"
CONFIG_DATABASE: Dict[str, Dict[str, Any]] = {
    "app.yaml": {
        "app_name": "MyApp",
        "version": "1.2.3",
        "logging": {"level": "INFO", "handlers": ["console", "file"]},
        "features": {"beta": True, "experiments": ["new-ui", "fast-path"]},
    },
    "db.json": {
        "engine": "postgresql",
        "host": "db.internal",
        "port": 5432,
        "database": "appdb",
        "user": "appuser",
        "pool": {"min": 2, "max": 10, "timeout_seconds": 30},
    },
    "feature.toml": {
        "flags": {"featureA": True, "featureB": False},
        "rollout": {"featureA": 100, "featureB": 25},
    },
}


def read_config_file(file_name: str) -> Dict[str, Any]:
    """
    Return configuration details for the given file name as a dictionary.
    Simulates file reads using CONFIG_DATABASE. Raises IOError if not found.
    """
    try:
        config = CONFIG_DATABASE[file_name]
    except KeyError as exc:
        raise IOError(f"Configuration file not found: {file_name}") from exc
    return deepcopy(config)
