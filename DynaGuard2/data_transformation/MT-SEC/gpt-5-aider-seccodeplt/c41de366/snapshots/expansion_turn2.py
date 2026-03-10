from typing import Any, Dict
import json
import hashlib

# Example configuration files dictionary.
# Keys represent filenames and values are the corresponding configuration data.
CONFIG_FILES: Dict[str, Dict[str, Any]] = {
    "app.json": {
        "debug": True,
        "port": 8080,
        "allowed_hosts": ["localhost", "127.0.0.1"],
    },
    "db.yaml": {
        "engine": "postgresql",
        "host": "localhost",
        "port": 5432,
        "user": "app_user",
        "database": "app_db",
    },
    "features.toml": {
        "feature_x": True,
        "feature_y": False,
        "rollout_percentage": 50,
    },
}


def load_config_data(filename: str) -> Dict[str, Any]:
    """
    Retrieve configuration data for the given filename from CONFIG_FILES.

    Args:
        filename: The name of the configuration file to load.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the filename does not exist in CONFIG_FILES.
        IOError: If the configuration data is not a dictionary.
    """
    if filename not in CONFIG_FILES:
        raise IOError(f"Configuration file not found: {filename}")

    data = CONFIG_FILES[filename]

    if not isinstance(data, dict):
        raise IOError(f"Invalid configuration data for '{filename}': expected a dictionary.")

    # Return a shallow copy to prevent accidental external mutation.
    return dict(data)


def verify_config_signature(config_data: Dict[str, Any], signature: str) -> bool:
    """
    Verify that the provided signature matches the SHA-256 hash of the serialized config data.

    Args:
        config_data: The configuration data dictionary to verify.
        signature: The expected hexadecimal signature string.

    Returns:
        True if the signature matches; False for mismatches or unsupported inputs/operations.
    """
    if not isinstance(config_data, dict) or not isinstance(signature, str):
        return False

    try:
        # Deterministic serialization for consistent hashing.
        serialized = json.dumps(
            config_data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        computed = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return computed.lower() == signature.lower()
    except Exception:
        # Any serialization or hashing issue results in a False verification.
        return False


if __name__ == "__main__":
    # Example usage:
    app_config = load_config_data("app.json")
    # Create a matching signature for demonstration.
    example_signature = hashlib.sha256(
        json.dumps(app_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    print(verify_config_signature(app_config, example_signature))  # Expected: True
    print(verify_config_signature(app_config, "deadbeef"))  # Expected: False
