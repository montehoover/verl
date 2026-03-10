from typing import Any, Dict


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
