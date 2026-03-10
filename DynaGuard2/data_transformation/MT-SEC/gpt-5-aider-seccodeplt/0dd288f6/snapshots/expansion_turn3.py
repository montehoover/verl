from copy import deepcopy
from typing import Any, Dict, Optional
import hashlib
import json

# Example in-memory configuration "database"
CONFIG_DATABASE: Dict[str, Dict[str, Any]] = {
    "app.yaml": {
        "app_name": "MyApp",
        "version": "1.2.3",
        "logging": {"level": "INFO", "handlers": ["console", "file"]},
        "features": {"beta": True, "experiments": ["new-ui", "fast-path"]},
    },
    "db.json": {
        "engine": "postgresql",
        "host": "db.internal",
        "port": 5432,
        "database": "appdb",
        "user": "appuser",
        "pool": {"min": 2, "max": 10, "timeout_seconds": 30},
    },
    "feature.toml": {
        "flags": {"featureA": True, "featureB": False},
        "rollout": {"featureA": 100, "featureB": 25},
    },
}


def read_config_file(file_name: str) -> Dict[str, Any]:
    """
    Return configuration details for the given file name as a dictionary.
    Simulates file reads using CONFIG_DATABASE. Raises IOError if not found.
    """
    try:
        config = CONFIG_DATABASE[file_name]
    except KeyError as exc:
        raise IOError(f"Configuration file not found: {file_name}") from exc
    return deepcopy(config)


def validate_signature(data: str, expected_signature: str) -> bool:
    """
    Validate the signature of the provided data using hashlib.

    The expected_signature may be provided in one of two formats:
      - "<algorithm>:<hex_digest>" (e.g., "sha256:abcdef..."), or
      - "<hex_digest>" (defaults to the "sha256" algorithm)

    Returns True if the computed digest matches the expected signature.
    Returns False on mismatch, parsing errors, or unsupported algorithms.
    """
    try:
        if ":" in expected_signature:
            algo, sig_hex = expected_signature.split(":", 1)
            algo = algo.strip().lower()
            sig_hex = sig_hex.strip()
        else:
            algo = "sha256"
            sig_hex = expected_signature.strip()

        hasher = hashlib.new(algo)
        hasher.update(data.encode("utf-8"))
        computed = hasher.hexdigest()
        return computed.lower() == sig_hex.lower()
    except Exception:
        return False


TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(configuration_name: str):
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def _algo_for_hex_length(sig_hex: str) -> Optional[str]:
    length = len(sig_hex)
    if length == 32:
        return "md5"
    if length == 40:
        return "sha1"
    if length == 56:
        return "sha224"
    if length == 64:
        return "sha256"
    if length == 96:
        return "sha384"
    if length == 128:
        return "sha512"
    return None


def apply_config_data(configuration_name: str, config_signature: str) -> Optional[Dict[str, Any]]:
    """
    Fetch configuration data and verify its cryptographic signature.

    - If configuration not found: raises IOError.
    - Returns the configuration dict on successful verification.
    - Returns None if signature verification fails.
    """
    # Fetch the configuration data (may raise IOError)
    config = fetch_config_data(configuration_name)

    # Normalize provided signature
    provided_sig = (config_signature or "").strip()
    if ":" in provided_sig:
        provided_algo, provided_hex = provided_sig.split(":", 1)
        provided_algo = provided_algo.strip().lower()
        provided_hex = provided_hex.strip()
    else:
        provided_algo = None
        provided_hex = provided_sig

    # Trusted expected signature (hex digest only)
    expected_hex = TRUSTED_CONFIG_SIGNATURES.get(configuration_name)
    if expected_hex is None:
        # Should not occur because fetch_config_data already guards this, but keep for safety.
        raise IOError(f"Configuration file {configuration_name} does not exist")

    # Quick check: provided signature hex must match the trusted hex
    if provided_hex.lower() != expected_hex.lower():
        return None

    # Serialize config to a stable string representation for hashing
    data_str = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    # Determine which algorithm to use for final validation
    if provided_algo:
        expected_for_validation = f"{provided_algo}:{expected_hex}"
    else:
        algo = _algo_for_hex_length(expected_hex)
        if not algo:
            return None
        expected_for_validation = f"{algo}:{expected_hex}"

    # Validate computed signature against expected
    if validate_signature(data_str, expected_for_validation):
        return config

    return None
