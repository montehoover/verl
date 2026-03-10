from typing import Any, Dict
from copy import deepcopy

# Example configuration database for demonstration purposes.
CONFIG_DATABASE: Dict[str, Dict[str, Any]] = {
    "default": {
        "database_url": "sqlite:///app.db",
        "log_level": "INFO",
        "feature_flags": {
            "beta_mode": False,
            "enable_metrics": True,
        },
        "retry_policy": {
            "retries": 3,
            "backoff_seconds": 2,
        },
    },
    "development": {
        "database_url": "postgresql://dev:dev@localhost:5432/app_dev",
        "log_level": "DEBUG",
        "feature_flags": {
            "beta_mode": True,
            "enable_metrics": True,
        },
        "retry_policy": {
            "retries": 1,
            "backoff_seconds": 1,
        },
    },
    "production": {
        "database_url": "postgresql://prod:secure@db.internal:5432/app",
        "log_level": "WARNING",
        "feature_flags": {
            "beta_mode": False,
            "enable_metrics": True,
        },
        "retry_policy": {
            "retries": 5,
            "backoff_seconds": 5,
        },
    },
}


def load_config_data(config_name: str) -> Dict[str, Any]:
    """
    Load configuration data from the local CONFIG_DATABASE by name.

    Args:
        config_name: The name of the configuration to retrieve.

    Returns:
        A deep-copied dictionary representing the configuration data.

    Raises:
        IOError: If the configuration name is not found.
    """
    try:
        data = CONFIG_DATABASE[config_name]
    except KeyError as exc:
        raise IOError(f"Configuration '{config_name}' not found.") from exc

    # Return a deep copy to prevent callers from mutating the global database.
    return deepcopy(data)


if __name__ == "__main__":
    # Simple demonstration:
    # This will print the configuration for "default".
    import json
    print(json.dumps(load_config_data("default"), indent=2, sort_keys=True))
