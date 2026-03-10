import json
import configparser
from io import StringIO

trusted_formats = ["json", "ini"]

def validate_format(serialization_format: str) -> None:
    """Validate that the serialization format is trusted."""
    if serialization_format not in trusted_formats:
        raise ValueError(f"Untrusted format: {serialization_format}. Only {trusted_formats} are allowed.")

def deserialize_json(byte_data: bytes) -> object:
    """Deserialize JSON data from bytes."""
    try:
        return json.loads(byte_data.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding: {e}")

def deserialize_ini(byte_data: bytes) -> dict:
    """Deserialize INI data from bytes."""
    try:
        config = configparser.ConfigParser()
        config.read_string(byte_data.decode('utf-8'))
        # Convert ConfigParser to dict
        result = {}
        for section in config.sections():
            result[section] = dict(config.items(section))
        return result
    except configparser.Error as e:
        raise ValueError(f"Invalid INI data: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding: {e}")

def securely_load_data(byte_data: bytes, serialization_format: str):
    validate_format(serialization_format)
    
    if serialization_format == "json":
        return deserialize_json(byte_data)
    elif serialization_format == "ini":
        return deserialize_ini(byte_data)
    else:
        raise NotImplementedError(f"Format {serialization_format} is not implemented")
