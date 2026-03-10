from typing import Any, Dict
import hashlib

# Example configuration database for demonstration purposes.
# In a real system, these would correspond to file contents parsed from disk.
CONFIG_DATABASE: Dict[str, Dict[str, Any]] = {
    "app.yaml": {
        "app": {
            "name": "DemoApp",
            "version": "1.0.0",
            "debug": True,
        }
    },
    "db.json": {
        "database": {
            "host": "localhost",
            "port": 5432,
            "user": "app_user",
            "password": "secret",
        }
    },
    "features.ini": {
        "features": {
            "signup": True,
            "beta_mode": False,
        }
    },
}


def fetch_config_data(filename: str) -> Dict[str, Any]:
    """
    Retrieve configuration data from the predefined local CONFIG_DATABASE.

    Args:
        filename: The name of the configuration "file" to fetch.

    Returns:
        The configuration data associated with the given filename.

    Raises:
        IOError: If the configuration file is not found in CONFIG_DATABASE.
    """
    try:
        return CONFIG_DATABASE[filename]
    except KeyError as exc:
        raise IOError(f"Configuration file '{filename}' not found in local database.") from exc


def verify_config_signature(config_data: str, signature: str) -> bool:
    """
    Verify that the given signature matches the SHA-256 hash of the config_data.

    Args:
        config_data: The configuration data as a string.
        signature: The expected signature (hex-encoded SHA-256 digest).

    Returns:
        True if the signature matches; False otherwise or on unsupported inputs/operations.
    """
    if not isinstance(config_data, str) or not isinstance(signature, str):
        return False

    try:
        computed_digest = hashlib.sha256(config_data.encode("utf-8")).hexdigest()
    except Exception:
        return False

    provided = signature.strip().lower()
    return computed_digest == provided
