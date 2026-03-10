from typing import Any, Dict

# Example configuration database for demonstration
CONFIG_DATABASE: Dict[str, Dict[str, Any]] = {
    "app_config.json": {
        "app_name": "MyApp",
        "version": "1.0.0",
        "debug": False,
        "features": {
            "analytics": True,
            "beta_mode": False
        }
    },
    "db_config.yaml": {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "user",
            "password": "pass",
            "name": "my_database"
        },
        "pool": {
            "min": 2,
            "max": 10
        }
    },
    "feature_flags.toml": {
        "flags": {
            "new_dashboard": True,
            "use_cache_v2": False,
            "enable_experimental_api": True
        }
    }
}


def get_config_data(filename: str) -> Dict[str, Any]:
    """
    Retrieve configuration data for the given filename from CONFIG_DATABASE.

    Args:
        filename: The name of the configuration file to fetch.

    Returns:
        The configuration data as a dictionary.

    Raises:
        IOError: If the configuration file is not found in CONFIG_DATABASE.
    """
    try:
        return CONFIG_DATABASE[filename]
    except KeyError as e:
        raise IOError(f"Configuration file '{filename}' not found") from e


if __name__ == "__main__":
    # Demonstration:
    # Successful fetch
    cfg1 = get_config_data("app_config.json")
    print("Loaded app_config.json:", cfg1)

    # Simulated not-found scenario (will raise IOError)
    try:
        get_config_data("missing_config.json")
    except IOError as err:
        print("Error:", err)
