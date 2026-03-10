import os

def get_environment_data(var_name: str) -> str:
    """
    Fetches configuration data from the local environment.

    Args:
        var_name: The name of the environment variable to fetch.

    Returns:
        The value of the environment variable as a string.

    Raises:
        ValueError: If the environment variable does not exist.
    """
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found.")
    return value
