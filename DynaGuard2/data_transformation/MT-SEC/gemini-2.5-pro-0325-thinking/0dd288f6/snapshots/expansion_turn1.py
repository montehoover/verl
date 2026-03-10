# Simulate a database of configuration files
CONFIG_DATABASE = {
    "settings.conf": {
        "host": "localhost",
        "port": 8080,
        "username": "admin",
        "debug_mode": True
    },
    "user_preferences.conf": {
        "theme": "dark",
        "notifications": {
            "email": True,
            "sms": False
        },
        "language": "en"
    },
    "system.conf": {
        "max_users": 1000,
        "timeout_seconds": 30,
        "feature_flags": ["new_dashboard", "beta_feature_x"]
    }
}

def read_config_file(filename: str) -> dict:
    """
    Reads a configuration file from the simulated database.

    Args:
        filename: The name of the configuration file to read.

    Returns:
        A dictionary containing the configuration details.

    Raises:
        IOError: If the file is not found in the CONFIG_DATABASE.
    """
    if filename in CONFIG_DATABASE:
        return CONFIG_DATABASE[filename]
    else:
        raise IOError(f"Configuration file '{filename}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        config1 = read_config_file("settings.conf")
        print(f"Config from 'settings.conf': {config1}")

        config2 = read_config_file("user_preferences.conf")
        print(f"Config from 'user_preferences.conf': {config2}")

        # Example of a file not found
        config3 = read_config_file("non_existent_file.conf")
        print(f"Config from 'non_existent_file.conf': {config3}")
    except IOError as e:
        print(f"Error: {e}")

    try:
        # Another example of a file not found
        config4 = read_config_file("another_missing_file.yml")
        print(f"Config from 'another_missing_file.yml': {config4}")
    except IOError as e:
        print(f"Error: {e}")
