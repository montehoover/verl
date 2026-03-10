import os

def get_environment_data(var_name):
    value = os.environ.get(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found")
    return value
