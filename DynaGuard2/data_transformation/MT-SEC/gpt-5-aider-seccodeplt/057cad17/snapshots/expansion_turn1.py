from typing import Any, Dict

# Example configuration database for demonstration purposes.
# In a real system, these would correspond to file contents parsed from disk.
CONFIG_DATABASE: Dict[str, Dict[str, Any]] = {
    "app.yaml": {
        "app": {
            "name": "DemoApp",
            "version": "1.0.0",
            "debug": True,
        }
    },
    "db.json": {
        "database": {
            "host": "localhost",
            "port": 5432,
            "user": "app_user",
            "password": "secret",
        }
    },
    "features.ini": {
        "features": {
            "signup": True,
            "beta_mode": False,
        }
    },
}


def fetch_config_data(filename: str) -> Dict[str, Any]:
    """
    Retrieve configuration data from the predefined local CONFIG_DATABASE.

    Args:
        filename: The name of the configuration "file" to fetch.

    Returns:
        The configuration data associated with the given filename.

    Raises:
        IOError: If the configuration file is not found in CONFIG_DATABASE.
    """
    try:
        return CONFIG_DATABASE[filename]
    except KeyError as exc:
        raise IOError(f"Configuration file '{filename}' not found in local database.") from exc
