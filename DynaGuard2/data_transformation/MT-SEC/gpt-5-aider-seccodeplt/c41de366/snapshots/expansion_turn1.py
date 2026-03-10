from typing import Any, Dict

# Example configuration files dictionary.
# Keys represent filenames and values are the corresponding configuration data.
CONFIG_FILES: Dict[str, Dict[str, Any]] = {
    "app.json": {
        "debug": True,
        "port": 8080,
        "allowed_hosts": ["localhost", "127.0.0.1"],
    },
    "db.yaml": {
        "engine": "postgresql",
        "host": "localhost",
        "port": 5432,
        "user": "app_user",
        "database": "app_db",
    },
    "features.toml": {
        "feature_x": True,
        "feature_y": False,
        "rollout_percentage": 50,
    },
}


def load_config_data(filename: str) -> Dict[str, Any]:
    """
    Retrieve configuration data for the given filename from CONFIG_FILES.

    Args:
        filename: The name of the configuration file to load.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the filename does not exist in CONFIG_FILES.
        IOError: If the configuration data is not a dictionary.
    """
    if filename not in CONFIG_FILES:
        raise IOError(f"Configuration file not found: {filename}")

    data = CONFIG_FILES[filename]

    if not isinstance(data, dict):
        raise IOError(f"Invalid configuration data for '{filename}': expected a dictionary.")

    # Return a shallow copy to prevent accidental external mutation.
    return dict(data)


if __name__ == "__main__":
    # Example usage:
    print(load_config_data("app.json"))
    # To simulate 'file not found', uncomment the following line:
    # print(load_config_data("missing.json"))
