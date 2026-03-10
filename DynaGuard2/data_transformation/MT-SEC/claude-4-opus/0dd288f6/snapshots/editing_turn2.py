import os

TRUSTED_VARIABLES = {}

def get_environment_data(var_name):
    if var_name not in TRUSTED_VARIABLES:
        return None
    
    value = os.environ.get(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found")
    return value
