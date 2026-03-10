from typing import Dict, Any

# Simulated local configuration storage with dummy data
_CONFIG_STORE: Dict[str, Dict[str, Any]] = {
    "development": {
        "debug": True,
        "database": "sqlite:///dev.db",
        "feature_flags": {
            "new_ui": True,
            "beta_feature": False,
        },
        "log_level": "DEBUG",
    },
    "production": {
        "debug": False,
        "database": "postgresql://user:pass@prod-db:5432/app",
        "feature_flags": {
            "new_ui": False,
            "beta_feature": False,
        },
        "log_level": "WARNING",
    },
    "testing": {
        "debug": True,
        "database": "sqlite:///:memory:",
        "feature_flags": {
            "new_ui": True,
            "beta_feature": True,
        },
        "log_level": "INFO",
    },
}


def load_local_config(config_name: str) -> Dict[str, Any]:
    """
    Retrieve configuration data by name from a local source.

    Args:
        config_name: The name of the configuration to load.

    Returns:
        A dictionary containing dummy configuration data if found;
        otherwise, an empty dictionary.
    """
    data = _CONFIG_STORE.get(config_name)
    # Return a shallow copy to prevent accidental mutations of the store
    return dict(data) if data is not None else {}
