from typing import Any, Dict
import hashlib

# Example configuration database for demonstration
CONFIG_DATABASE: Dict[str, Dict[str, Any]] = {
    "app_config.json": {
        "app_name": "MyApp",
        "version": "1.0.0",
        "debug": False,
        "features": {
            "analytics": True,
            "beta_mode": False
        }
    },
    "db_config.yaml": {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "user",
            "password": "pass",
            "name": "my_database"
        },
        "pool": {
            "min": 2,
            "max": 10
        }
    },
    "feature_flags.toml": {
        "flags": {
            "new_dashboard": True,
            "use_cache_v2": False,
            "enable_experimental_api": True
        }
    }
}


def get_config_data(filename: str) -> Dict[str, Any]:
    """
    Retrieve configuration data for the given filename from CONFIG_DATABASE.

    Args:
        filename: The name of the configuration file to fetch.

    Returns:
        The configuration data as a dictionary.

    Raises:
        IOError: If the configuration file is not found in CONFIG_DATABASE.
    """
    try:
        return CONFIG_DATABASE[filename]
    except KeyError as e:
        raise IOError(f"Configuration file '{filename}' not found") from e


def validate_signature(data: str, signature: str) -> bool:
    """
    Validate that the given signature matches the SHA-256 hex digest of the provided data.

    Returns:
        True if the signature matches; False for mismatches or on any unsupported inputs/operations.
    """
    # Basic input validation
    if not isinstance(data, str) or not isinstance(signature, str):
        return False

    try:
        # Compute SHA-256 hash of the data
        digest = hashlib.sha256(data.encode("utf-8")).hexdigest()

        # Validate signature shape (must be 64 hex chars for SHA-256)
        if len(signature) != len(digest):
            return False

        # Ensure signature is valid hexadecimal
        try:
            int(signature, 16)
        except ValueError:
            return False

        # Case-insensitive comparison of hex digests
        return digest == signature.lower()
    except Exception:
        # Any unexpected error is treated as unsupported/invalid input
        return False


if __name__ == "__main__":
    # Demonstration:
    # Successful fetch
    cfg1 = get_config_data("app_config.json")
    print("Loaded app_config.json:", cfg1)

    # Simulated not-found scenario (will raise IOError)
    try:
        get_config_data("missing_config.json")
    except IOError as err:
        print("Error:", err)

    # Signature validation demo
    data_example = "hello world"
    good_sig = hashlib.sha256(data_example.encode("utf-8")).hexdigest()
    bad_sig = "0" * 64

    print("Valid signature:", validate_signature(data_example, good_sig))  # True
    print("Invalid signature:", validate_signature(data_example, bad_sig))  # False
    print("Unsupported input:", validate_signature(123, good_sig))  # False due to non-string data
