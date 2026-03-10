import os

TRUSTED_VARIABLES = {}

def get_environment_data(var_name):
    if var_name not in TRUSTED_VARIABLES:
        return None
    
    value = os.environ.get(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found")
    return value

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(configuration_name: str):
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def apply_config_data(configuration_name, config_signature):
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        return None
    
    if TRUSTED_CONFIG_SIGNATURES[configuration_name] != config_signature:
        return None
    
    return fetch_config_data(configuration_name)
