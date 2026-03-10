from typing import Any, Dict
from copy import deepcopy
import hashlib

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


if __name__ == "__main__":
    # Simple demonstration:
    # This will print the configuration for "default".
    import json
    print(json.dumps(load_config_data("default"), indent=2, sort_keys=True))
