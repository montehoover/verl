import json
import configparser
import pickle
from io import StringIO

safe_formats = ["json", "ini"]

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
    if struct_format not in safe_formats:
        raise ValueError(f"Format '{struct_format}' is not recognized or deemed unsafe")
    
    if struct_format == "json":
        try:
            return json.loads(byte_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    
    elif struct_format == "ini":
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
