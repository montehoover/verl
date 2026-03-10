import json
import configparser

valid_formats = ["json", "ini"]

def convert_serialized_data(raw_bytes: bytes, format_hint: str):
    if format_hint not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint}")
    
    if format_hint == "json":
        try:
            return json.loads(raw_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    
    elif format_hint == "ini":
        try:
            config = configparser.ConfigParser()
            config.read_string(raw_bytes.decode('utf-8'))
            # Convert ConfigParser to dict
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            return result
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize INI data: {e}")
