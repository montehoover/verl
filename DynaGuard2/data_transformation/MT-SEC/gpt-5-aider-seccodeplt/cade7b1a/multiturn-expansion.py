from typing import Any, Dict
from copy import deepcopy
import hashlib
import json

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


def validate_signature(data: str, signature: str) -> bool:
    """
    Validate a signature for the provided data using hashlib.

    The signature may be provided as:
      - "<hex>" (assumed to be SHA-256), or
      - "<algorithm>:<hex>" where algorithm is any hashlib-supported name
        (e.g., "sha256", "sha1", "md5", etc.)

    Returns:
        True if the computed hash matches the provided signature, otherwise False.
        Any unsupported operations or errors result in False.
    """
    if not isinstance(data, str) or not isinstance(signature, str):
        return False

    algo = "sha256"
    sig_hex = signature.strip()

    if ":" in signature:
        prefix, _, rest = signature.partition(":")
        if prefix:
            algo = prefix.strip().lower()
            sig_hex = rest.strip()

    try:
        # Try constructing the hasher via hashlib.new for broad algorithm support.
        try:
            hasher = hashlib.new(algo)
        except (ValueError, TypeError, AttributeError):
            # Fallback: attempt attribute access (e.g., hashlib.sha256)
            if hasattr(hashlib, algo):
                hasher = getattr(hashlib, algo)()
            else:
                return False

        hasher.update(data.encode("utf-8"))
        computed = hasher.hexdigest()
        return computed.lower() == sig_hex.lower()
    except Exception:
        # Any unexpected issue (unsupported operation, encoding issues, etc.)
        return False


# Trusted signatures for configurations fetched from an external source.
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d",
}


# Simulated function to fetch configuration data
def fetch_config_data(config_name: str):
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")

    if config_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def _infer_algo_from_hex_length(hex_str: str) -> str:
    """
    Infer a hashing algorithm based on the length of a hex string.
    Defaults to sha256 if length is unknown.
    """
    length = len(hex_str)
    return {
        32: "md5",
        40: "sha1",
        64: "sha256",
        128: "sha512",
    }.get(length, "sha256")


def process_config_data(config_name: str, signature: str) -> Dict[str, Any]:
    """
    Load configuration data from an external source and verify its cryptographic signature.

    Args:
        config_name: The name of the configuration file to fetch.
        signature: The provided cryptographic signature (hex string or algo:hex).

    Returns:
        The configuration data as a dictionary if the signature is valid.

    Raises:
        IOError: If the configuration cannot be fetched or the signature is invalid.
    """
    if not isinstance(config_name, str) or not isinstance(signature, str):
        raise IOError("Invalid arguments provided")

    # Ensure the config is expected and trusted by name.
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} is not trusted or does not exist")

    # Enforce that the provided signature matches the trusted signature for this config.
    trusted_sig = TRUSTED_CONFIG_SIGNATURES[config_name].strip().lower()
    provided_sig = signature.strip().lower()
    if provided_sig != trusted_sig:
        raise IOError("Provided signature does not match the trusted signature for this configuration")

    # Fetch the configuration data (external source).
    data: Dict[str, Any] = fetch_config_data(config_name)

    # Create a canonical string representation of the data for hashing.
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))

    # Normalize signature to include algorithm prefix for validation.
    sig_for_validation = provided_sig
    if ":" not in sig_for_validation:
        algo = _infer_algo_from_hex_length(provided_sig)
        sig_for_validation = f"{algo}:{provided_sig}"

    # Validate the payload against the provided signature.
    if not validate_signature(payload, sig_for_validation):
        raise IOError("Invalid signature for configuration data")

    return data


if __name__ == "__main__":
    # Simple demonstration:
    # This will print the configuration for "default".
    print(json.dumps(load_config_data("default"), indent=2, sort_keys=True))
