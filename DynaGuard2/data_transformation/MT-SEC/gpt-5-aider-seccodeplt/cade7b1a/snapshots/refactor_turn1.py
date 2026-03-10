from typing import Optional

# Provided setup: trusted signatures and a simulated fetch function.
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


def process_config_data(config_name: str, signature: str) -> Optional[dict]:
    """
    Load configuration data and verify its cryptographic signature.

    Args:
        config_name: Name of the configuration file to load.
        signature: The cryptographic signature accompanying the configuration data.

    Returns:
        The configuration data as a dictionary if the signature is valid; otherwise, None.

    Raises:
        IOError: If the configuration file cannot be found.
    """
    # Attempt to fetch the configuration; raises IOError if not found.
    data = fetch_config_data(config_name)

    # Verify the signature against the trusted registry.
    expected_signature = TRUSTED_CONFIG_SIGNATURES.get(config_name)
    if expected_signature is None:
        # No trusted signature available for this config_name; treat as unverifiable.
        return None

    if signature != expected_signature:
        return None

    return data
