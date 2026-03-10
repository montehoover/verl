import json
import configparser
import io

trusted_formats = ["json", "ini"]

def decode_serialized_data(data_bytes, format_string):
    if format_string not in trusted_formats:
        raise ValueError(f"Untrusted format: {format_string}")
    
    if format_string == "json":
        try:
            return json.loads(data_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode JSON data: {e}")
    
    elif format_string == "ini":
        try:
            config = configparser.ConfigParser()
            config.read_string(data_bytes.decode('utf-8'))
            # Convert ConfigParser to dict
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            return result
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode INI data: {e}")
    
    else:
        raise ValueError(f"Unsupported format: {format_string}")
