from config_utils import fetch_config_data

def handle_config_file(file_name):
    try:
        config_data = fetch_config_data(file_name)
        return config_data
    except FileNotFoundError:
        raise IOError(f"Configuration file '{file_name}' not found")
