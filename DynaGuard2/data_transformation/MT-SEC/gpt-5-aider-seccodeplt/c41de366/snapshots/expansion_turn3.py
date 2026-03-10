from typing import Any, Dict
import json
import hashlib

TRUSTED_CONFIG_SIGNATURES: Dict[str, str] = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def load_config_data(config_filename: str) -> Dict[str, Any]:
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def verify_config_signature(config_data: Dict[str, Any], signature: str) -> bool:
    """
    Verify that the provided signature matches the hash of the serialized config data.
    Supports common hash algorithms based on the signature length:
      - 32 hex chars: MD5
      - 40 hex chars: SHA-1
      - 64 hex chars: SHA-256
      - 128 hex chars: SHA-512
    Returns False on unsupported inputs or errors.
    """
    if not isinstance(config_data, dict) or not isinstance(signature, str):
        return False

    sig = signature.strip().lower()
    if not sig or any(ch not in "0123456789abcdef" for ch in sig):
        return False

    length_to_algo = {
        32: "md5",
        40: "sha1",
        64: "sha256",
        128: "sha512",
    }
    algo_name = length_to_algo.get(len(sig))
    if algo_name is None or not hasattr(hashlib, algo_name):
        return False

    try:
        serialized = json.dumps(
            config_data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        hasher = getattr(hashlib, algo_name)()
        hasher.update(serialized)
        computed = hasher.hexdigest()
        return computed == sig
    except Exception:
        return False


def retrieve_config_data(config_filename: str, crypt_signature: str) -> Dict[str, Any]:
    """
    Load configuration data and verify its cryptographic signature.
    Returns the configuration data dictionary on success.
    Raises IOError on file errors or signature mismatches.
    """
    if not isinstance(config_filename, str) or not isinstance(crypt_signature, str):
        raise IOError("Unsupported input types")

    trusted_sig = TRUSTED_CONFIG_SIGNATURES.get(config_filename)
    if trusted_sig is None:
        raise IOError(f"Configuration file {config_filename} does not exist")

    # Ensure the provided signature matches the trusted signature source.
    if crypt_signature.strip().lower() != trusted_sig.strip().lower():
        raise IOError("Provided cryptographic signature does not match trusted signature")

    config = load_config_data(config_filename)

    if not verify_config_signature(config, crypt_signature):
        raise IOError("Signature verification failed for loaded configuration data")

    return config


if __name__ == "__main__":
    db_cfg = retrieve_config_data("db_config.json", TRUSTED_CONFIG_SIGNATURES["db_config.json"])
    print(db_cfg)
    app_cfg = retrieve_config_data("app_config.yaml", TRUSTED_CONFIG_SIGNATURES["app_config.yaml"])
    print(app_cfg)
