from config_utils import fetch_config_data

TRUSTED_CONFIG_SIGNATURES = {}

def handle_config_file(file_name, sign):
    try:
        if file_name in TRUSTED_CONFIG_SIGNATURES and TRUSTED_CONFIG_SIGNATURES[file_name] == sign:
            config_data = fetch_config_data(file_name)
            return config_data
        else:
            return None
    except IOError:
        raise IOError(f"Configuration file {file_name} does not exist")
