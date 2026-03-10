import json
import configparser

trusted_formats = ["json", "ini"]

def securely_load_data(byte_data, serialization_format):
    if serialization_format not in trusted_formats:
        raise ValueError(f"Unsupported or insecure format: {serialization_format}")
    
    if serialization_format == 'json':
        return json.loads(byte_data.decode('utf-8'))
    elif serialization_format == 'ini':
        config = configparser.ConfigParser()
        config.read_string(byte_data.decode('utf-8'))
        return {section: dict(config.items(section)) for section in config.sections()}
