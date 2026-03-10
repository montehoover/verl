import json
import configparser

trusted_formats = ["json", "ini"]

def decode_serialized_data(data_bytes, format_string):
    if format_string not in trusted_formats:
        raise ValueError(f"Untrusted format: {format_string}")
    
    if format_string == "json":
        return json.loads(data_bytes.decode())
    elif format_string == "ini":
        config = configparser.ConfigParser()
        config.read_string(data_bytes.decode())
        return {section: dict(config.items(section)) for section in config.sections()}
