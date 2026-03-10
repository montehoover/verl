CONFIG_DATABASE = {
    "development": {
        "debug": True,
        "database_uri": "sqlite:///dev.db",
        "secret_key": "dev_secret"
    },
    "production": {
        "debug": False,
        "database_uri": "postgresql://user:pass@host:port/dbname",
        "secret_key": "prod_secret_very_secure"
    },
    "testing": {
        "debug": True,
        "database_uri": "sqlite:///:memory:",
        "secret_key": "test_secret",
        "testing_mode": True
    }
}

def load_config_data(config_name: str) -> dict:
    """
    Retrieves configuration data from a predefined local dictionary.

    Args:
        config_name: The name of the configuration to load.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the configuration name is not found in the database.
    """
    if config_name in CONFIG_DATABASE:
        return CONFIG_DATABASE[config_name]
    else:
        raise IOError(f"Configuration '{config_name}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        dev_config = load_config_data("development")
        print("Development Config:", dev_config)

        prod_config = load_config_data("production")
        print("Production Config:", prod_config)

        # Example of a missing configuration
        missing_config = load_config_data("staging")
        print("Staging Config:", missing_config)
    except IOError as e:
        print(f"Error: {e}")

    try:
        test_config = load_config_data("testing")
        print("Testing Config:", test_config)
    except IOError as e:
        print(f"Error: {e}")
