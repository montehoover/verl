from typing import Any, Dict
import hashlib
import json


# Sample configuration setup (in-memory "database" of configuration files)
CONFIG_DATABASE: Dict[str, Any] = {
    "app.json": {
        "feature_flags": {"enable_signup": True, "beta_mode": False},
        "theme": "dark",
        "version": "1.2.3",
    },
    "database.yaml": {
        "host": "localhost",
        "port": 5432,
        "user": "app_user",
        "password": "change_me",
        "pool_size": 10,
    },
    "logging.toml": {
        "level": "INFO",
        "handlers": ["console", "file"],
        "file": {"path": "/var/log/app.log", "rotate": True},
    },
}


def read_config_file(filename: str) -> Any:
    """
    Retrieve configuration data given a filename.

    This function simulates reading from an external configuration file by
    looking up the content in the predefined CONFIG_DATABASE dictionary.

    Args:
        filename: The name of the configuration "file" to retrieve.

    Returns:
        The configuration data associated with the given filename.

    Raises:
        IOError: If the filename is not found in CONFIG_DATABASE.
    """
    try:
        return CONFIG_DATABASE[filename]
    except KeyError as exc:
        raise IOError(f"Configuration file not found: {filename}") from exc


def validate_signature(data: str, signature: str) -> bool:
    """
    Verify a digital signature by hashing the provided data and comparing it to the signature.

    Args:
        data: The original data as a string.
        signature: The expected signature (hex string) to compare against.

    Returns:
        True if the computed hash matches the provided signature; otherwise False.
        Returns False if any exception occurs during processing.
    """
    try:
        if not isinstance(data, str) or not isinstance(signature, str):
            return False
        computed = hashlib.sha256(data.encode("utf-8")).hexdigest()
        return computed.lower() == signature.strip().lower()
    except Exception:
        return False


TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(file_name: str):
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} does not exist")

    if file_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif file_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def handle_config_file(file_name: str, sign: str):
    """
    Fetch configuration data and verify its signature.

    Args:
        file_name: The name of the configuration file to fetch.
        sign: The signature string to validate against the fetched data.

    Returns:
        The configuration data (e.g., dict) if signature validation succeeds; otherwise None.
        Returns None if any error occurs (e.g., missing file or validation error).
    """
    try:
        data = fetch_config_data(file_name)

        # Canonicalize the data to a stable string for hashing
        if isinstance(data, (dict, list)):
            payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        else:
            payload = str(data)

        if validate_signature(payload, sign):
            return data
        return None
    except Exception:
        return None


if __name__ == "__main__":
    # Demonstration of reading configuration files
    print("Available configuration files:", ", ".join(CONFIG_DATABASE.keys()))

    print("\nReading 'app.json':")
    app_config = read_config_file("app.json")
    print(app_config)

    print("\nReading 'database.yaml':")
    db_config = read_config_file("database.yaml")
    print(db_config)

    print("\nAttempting to read a non-existent file 'missing.ini':")
    try:
        read_config_file("missing.ini")
    except IOError as e:
        print(f"Caught expected error: {e}")

    # Demonstration of signature validation
    print("\nSignature validation demo:")
    demo_data = "important configuration payload"
    valid_sig = hashlib.sha256(demo_data.encode("utf-8")).hexdigest()
    invalid_sig = "deadbeef"

    print("Valid signature check:", validate_signature(demo_data, valid_sig))
    print("Invalid signature check:", validate_signature(demo_data, invalid_sig))

    # Demonstration of handle_config_file
    print("\nhandle_config_file demo:")
    try:
        db_data = fetch_config_data("db_config.json")
        db_payload = json.dumps(db_data, sort_keys=True, separators=(",", ":"))
        db_valid_sig = hashlib.sha256(db_payload.encode("utf-8")).hexdigest()

        print("db_config.json with valid signature ->",
              handle_config_file("db_config.json", db_valid_sig))
        print("db_config.json with invalid signature ->",
              handle_config_file("db_config.json", "deadbeef"))
    except IOError as e:
        print(f"Error fetching demo config: {e}")
