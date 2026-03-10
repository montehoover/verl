import os
from typing import Optional

TRUSTED_VARIABLES: dict[str, bool] = {}

def get_environment_data(var_name: str) -> Optional[str]:
    """
    Fetch a configuration value from the local environment.

    Args:
        var_name: The name of the environment variable to retrieve.

    Returns:
        The configuration value as a string if the name is trusted, otherwise None.

    Raises:
        ValueError: If var_name is empty/whitespace or the environment variable does not exist or is empty.
    """
    if not isinstance(var_name, str) or not var_name.strip():
        raise ValueError("Environment variable name must be a non-empty string.")

    if var_name not in TRUSTED_VARIABLES:
        return None

    value = os.environ.get(var_name)
    if value is None or value == "":
        raise ValueError(f"Configuration value for '{var_name}' not found in environment.")

    return value
