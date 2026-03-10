def load_local_config(config_name: str) -> dict:
    """
    Retrieves configuration data from a local source.

    Args:
        config_name: The name of the configuration to retrieve.

    Returns:
        A dictionary with configuration data if found, otherwise an empty dictionary.
    """
    if config_name == "dummy_config":
        return {
            "setting1": "value1",
            "setting2": 123,
            "feature_enabled": True
        }
    elif config_name == "another_config":
        return {
            "host": "localhost",
            "port": 8080
        }
    else:
        return {}
