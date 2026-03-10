def load_local_config(config_name):
    """
    Retrieves configuration data from a local source.
    
    Args:
        config_name (str): The name of the configuration to retrieve.
    
    Returns:
        dict: Configuration data if found, empty dictionary otherwise.
    """
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
