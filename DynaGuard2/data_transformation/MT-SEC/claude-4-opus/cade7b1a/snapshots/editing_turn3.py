TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

def fetch_config_data(config_name: str):
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")

    if config_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def process_config_data(config_name: str, signature: str):
    """
    Loads configuration data from an external source and verifies it using cryptographic signatures.
    
    Args:
        config_name (str): The name of the configuration file to load.
        signature (str): The cryptographic signature.
    
    Returns:
        dict: Configuration data if the signature is valid, None otherwise.
    """
    # Check if the provided signature matches the trusted signature
    if config_name in TRUSTED_CONFIG_SIGNATURES:
        if TRUSTED_CONFIG_SIGNATURES[config_name] == signature:
            try:
                return fetch_config_data(config_name)
            except IOError:
                return None
    
    return None

def load_local_config(config_name, approved_configs):
    """
    Retrieves configuration data from a local source.
    
    Args:
        config_name (str): The name of the configuration to retrieve.
        approved_configs (list): List of approved configuration names.
    
    Returns:
        dict: Configuration data if found and approved, empty dictionary otherwise.
    """
    # Check if config_name is in the approved list
    if config_name not in approved_configs:
        return {}
    
    # Dummy configuration data
    configs = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "admin",
            "password": "secret123",
            "database_name": "myapp_db"
        },
        "api": {
            "base_url": "https://api.example.com",
            "version": "v1",
            "timeout": 30,
            "retry_count": 3
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "app.log",
            "max_size": "10MB"
        },
        "cache": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "ttl": 3600
        }
    }
    
    return configs.get(config_name, {})
