from typing import Any, Dict
import hashlib
import json

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
    Validate that the given signature matches a cryptographic hex digest of the provided data.
    Supports MD5 (32 hex chars) and SHA-256 (64 hex chars).

    Returns:
        True if the signature matches; False for mismatches or on any unsupported inputs/operations.
    """
    # Basic input validation
    if not isinstance(data, str) or not isinstance(signature, str):
        return False

    try:
        sig = signature.strip().lower()

        # Ensure signature is valid hexadecimal
        try:
            int(sig, 16)
        except ValueError:
            return False

        # Choose algorithm based on signature length
        if len(sig) == 32:
            digest = hashlib.md5(data.encode("utf-8")).hexdigest()
        elif len(sig) == 64:
            digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
        else:
            return False

        # Case-insensitive comparison of hex digests
        return digest == sig
    except Exception:
        # Any unexpected error is treated as unsupported/invalid input
        return False


# Trusted signatures mapping for externally fetched configurations
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def evaluate_config_file(config_filename: str, provided_sig: str) -> Dict[str, Any]:
    """
    Securely fetch and validate a configuration file by verifying its cryptographic signature.

    Args:
        config_filename: The configuration filename to fetch.
        provided_sig: The signature provided for verification.

    Returns:
        The configuration data as a dictionary upon successful verification.

    Raises:
        IOError: If the configuration file is missing or the signature verification fails.
        ValueError: If input types are invalid.
    """
    if not isinstance(config_filename, str) or not isinstance(provided_sig, str):
        raise ValueError("Invalid input types for evaluate_config_file")

    # Fetch the configuration data from the external source
    config_data = fetch_config_data(config_filename)

    # Compare the provided signature against the trusted one for this filename
    trusted_sig = TRUSTED_CONFIG_SIGNATURES.get(config_filename)
    if trusted_sig is None or provided_sig.strip().lower() != trusted_sig.strip().lower():
        raise IOError("Provided signature does not match trusted signature for the specified configuration")

    # Serialize configuration deterministically for hashing
    try:
        serialized = json.dumps(config_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:
        raise IOError("Failed to serialize configuration for signature verification") from e

    # Verify the signature against the serialized configuration content
    if not validate_signature(serialized, provided_sig):
        raise IOError("Signature verification failed for configuration content")

    return config_data


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
