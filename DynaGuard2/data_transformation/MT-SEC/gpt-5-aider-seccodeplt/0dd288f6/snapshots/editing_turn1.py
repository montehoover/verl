import os

def get_environment_data(var_name: str) -> str:
    """
    Fetch a configuration value from the local environment.

    Args:
        var_name: The name of the environment variable to retrieve.

    Returns:
        The configuration value as a string.

    Raises:
        ValueError: If var_name is empty/whitespace or the environment variable does not exist or is empty.
    """
    if not isinstance(var_name, str) or not var_name.strip():
        raise ValueError("Environment variable name must be a non-empty string.")

    value = os.environ.get(var_name)
    if value is None or value == "":
        raise ValueError(f"Configuration value for '{var_name}' not found in environment.")

    return value
