from typing import Dict, Any, List, Optional
import hmac

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


def load_local_config(approved_configs: List[str], config_name: str) -> Dict[str, Any]:
    """
    Retrieve configuration data by name from a local source, restricted to an approved list.

    Args:
        approved_configs: A list of approved configuration names that are allowed to be loaded.
        config_name: The name of the configuration to load.

    Returns:
        A dictionary containing dummy configuration data if the name is approved and found;
        otherwise, an empty dictionary.
    """
    if config_name not in approved_configs:
        return {}

    data = _CONFIG_STORE.get(config_name)
    # Return a shallow copy to prevent accidental mutations of the store
    return dict(data) if data is not None else {}


TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}


def fetch_config_data(config_name: str):
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")

    if config_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def process_config_data(config_name: str, signature: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration data from an external source and verify it using cryptographic signatures.

    Args:
        config_name: The name of the configuration file to load.
        signature: The cryptographic signature to verify.

    Returns:
        The configuration data as a dictionary if the signature is valid; otherwise, None.
    """
    expected_signature = TRUSTED_CONFIG_SIGNATURES.get(config_name)
    if expected_signature is None:
        return None

    # Use constant-time comparison for signature verification
    if not hmac.compare_digest(expected_signature, signature):
        return None

    try:
        data = fetch_config_data(config_name)
    except IOError:
        return None

    return dict(data)
