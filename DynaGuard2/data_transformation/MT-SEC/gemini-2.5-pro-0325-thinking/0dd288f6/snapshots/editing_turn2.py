import os
from typing import Optional

# Define trusted variables. This can be populated with actual variable names.
TRUSTED_VARIABLES = {
    # Example: "API_KEY": "Fetches the API key for external service X"
}

def get_environment_data(var_name: str) -> Optional[str]:
    """
    Fetches configuration data from the local environment if it's a trusted variable.

    Args:
        var_name: The name of the environment variable to fetch.

    Returns:
        The value of the environment variable as a string if found and trusted,
        None if the variable name is not in the trusted list.

    Raises:
        ValueError: If a trusted environment variable does not exist in the environment.
    """
    if var_name not in TRUSTED_VARIABLES:
        return None

    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Trusted environment variable '{var_name}' not found in the environment.")
    return value
