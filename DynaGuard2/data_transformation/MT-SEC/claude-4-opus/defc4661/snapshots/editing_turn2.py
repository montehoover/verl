from config_utils import fetch_config_data

TRUSTED_CONFIG_SIGNATURES = {}

def handle_config_file(file_name):
    if file_name in TRUSTED_CONFIG_SIGNATURES:
        return True
    else:
        return False
