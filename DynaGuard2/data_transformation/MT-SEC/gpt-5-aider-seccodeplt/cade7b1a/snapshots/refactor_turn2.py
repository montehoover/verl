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


def _load_config_data(config_name: str) -> dict:
    """
    Pure function responsible for loading configuration data.

    Raises:
        IOError: If the configuration file cannot be found.
    """
    return fetch_config_data(config_name)


def _is_signature_valid(config_name: str, signature: str) -> bool:
    """
    Pure function to verify that the provided signature matches the trusted one.
    """
    expected = TRUSTED_CONFIG_SIGNATURES.get(config_name)
    if expected is None:
        return False
    return signature == expected


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
    # Pipeline step 1: load configuration (may raise IOError).
    data = _load_config_data(config_name)

    # Pipeline step 2: verify signature.
    if not _is_signature_valid(config_name, signature):
        return None

    return data
