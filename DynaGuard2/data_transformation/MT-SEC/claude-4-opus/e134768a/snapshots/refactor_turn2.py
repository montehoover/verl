import json
import configparser
import pickle
from io import StringIO

safe_formats = ["json", "ini"]

def check_format_safety(struct_format: str) -> None:
    """
    Checks if the provided format is safe for deserialization.
    
    Args:
        struct_format: str, the format to check.
    
    Raises:
        ValueError: If the format is not recognized or deemed unsafe.
    """
    if struct_format not in safe_formats:
        raise ValueError(f"Format '{struct_format}' is not recognized or deemed unsafe")

def deserialize_json(byte_data: bytes):
    """
    Deserializes JSON data from bytes.
    
    Args:
        byte_data: bytes, the JSON data in binary form.
    
    Returns:
        The deserialized Python object.
    
    Raises:
        ValueError: If deserialization fails.
    """
    try:
        return json.loads(byte_data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to deserialize JSON data: {e}")

def deserialize_ini(byte_data: bytes):
    """
    Deserializes INI data from bytes.
    
    Args:
        byte_data: bytes, the INI data in binary form.
    
    Returns:
        The deserialized Python object as a dictionary.
    
    Raises:
        ValueError: If deserialization fails.
    """
    try:
        config = configparser.ConfigParser()
        config.read_string(byte_data.decode('utf-8'))
        # Convert ConfigParser to dictionary
        result = {}
        for section in config.sections():
            result[section] = dict(config.items(section))
        return result
    except (configparser.Error, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to deserialize INI data: {e}")

def unpack_serialized_object(byte_data: bytes, struct_format: str):
    """
    Deserializes binary data from potentially untrusted sources.
    
    Args:
        byte_data: bytes, the provided serialized data in a binary form.
        struct_format: str, the format used for deserialization (e.g., 'json', 'ini', 'pickle').
    
    Returns:
        The deserialized Python object contained within the data.
    
    Raises:
        ValueError: Must raise an error for formats not recognized or deemed unsafe.
    """
    check_format_safety(struct_format)
    
    if struct_format == "json":
        return deserialize_json(byte_data)
    elif struct_format == "ini":
        return deserialize_ini(byte_data)
