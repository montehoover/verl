import json
import configparser
import io

trusted_formats = ["json", "ini"]

def _deserialize_json(data_bytes):
    """Deserialize JSON data from bytes."""
    try:
        return json.loads(data_bytes.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to decode JSON data: {e}")

def _deserialize_ini(data_bytes):
    """Deserialize INI data from bytes."""
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

def decode_serialized_data(data_bytes, format_string):
    if format_string not in trusted_formats:
        raise ValueError(f"Untrusted format: {format_string}")
    
    deserializers = {
        "json": _deserialize_json,
        "ini": _deserialize_ini
    }
    
    deserializer = deserializers.get(format_string)
    if deserializer is None:
        raise ValueError(f"Unsupported format: {format_string}")
    
    return deserializer(data_bytes)
