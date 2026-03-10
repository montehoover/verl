from typing import Any, Dict
import hashlib
import json

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
    Verify that the given signature matches a supported hash of the config_data.

    Supported algorithms are inferred by signature length:
    - 32 hex chars: MD5
    - 64 hex chars: SHA-256

    Args:
        config_data: The configuration data as a string.
        signature: The expected signature (hex-encoded digest).

    Returns:
        True if the signature matches; False otherwise or on unsupported inputs/operations.
    """
    if not isinstance(config_data, str) or not isinstance(signature, str):
        return False

    provided = signature.strip().lower()

    # Validate hex and determine algorithm
    try:
        int(provided, 16)
    except ValueError:
        return False

    if len(provided) == 32:
        algo = "md5"
    elif len(provided) == 64:
        algo = "sha256"
    else:
        return False

    try:
        hasher = hashlib.new(algo)
        hasher.update(config_data.encode("utf-8"))
        computed_digest = hasher.hexdigest()
    except Exception:
        return False

    return computed_digest == provided


# Trusted signatures for specific external configuration files
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d",
}


# Simulated function to fetch configuration data from an external source
def external_fetch_config_data(filename: str):
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    if filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def handle_config_file(filename: str, sig: str):
    """
    Fetch configuration data from an external source and verify its cryptographic signature.

    The function ensures:
      - The filename is known and has a trusted signature.
      - The provided signature matches the trusted signature for the filename.
      - The provided signature matches the computed digest of the fetched configuration data.

    On success, returns the configuration data (dict). Raises IOError for missing files and
    ValueError for signature-related issues.
    """
    if not isinstance(filename, str) or not isinstance(sig, str):
        raise ValueError("Invalid inputs for filename/signature")

    trusted_sig = TRUSTED_CONFIG_SIGNATURES.get(filename)
    if trusted_sig is None:
        # Mirror the behavior of the external fetch: file doesn't exist
        raise IOError(f"Configuration file {filename} does not exist")

    # Ensure the provided signature is the trusted one for the file
    if trusted_sig.lower() != sig.strip().lower():
        raise ValueError("Untrusted signature for the specified configuration file")

    # Fetch the configuration from the external source
    config_obj = external_fetch_config_data(filename)

    # Convert configuration to a canonical string representation for hashing
    try:
        config_str = json.dumps(config_obj, sort_keys=True, separators=(",", ":"))
    except Exception as exc:
        raise ValueError("Failed to serialize configuration data for verification") from exc

    # Verify cryptographic signature matches the content
    if not verify_config_signature(config_str, sig):
        raise ValueError("Signature verification failed for configuration content")

    return config_obj
