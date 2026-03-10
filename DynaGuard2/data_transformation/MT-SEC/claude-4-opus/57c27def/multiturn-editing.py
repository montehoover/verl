import json
import configparser
import io

valid_formats = ["json", "ini"]

def convert_serialized_data(raw_bytes, format_hint):
    """
    Convert serialized data to a Python object based on format hint.
    
    Args:
        raw_bytes (bytes): The serialized data in bytes
        format_hint (str): The format type ('json', 'ini', 'pickle')
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the format is not supported or data is invalid
    """
    if format_hint not in valid_formats:
        raise ValueError(f"Unsupported format: {format_hint}. Supported formats are: {', '.join(valid_formats)}")
    
    if format_hint == 'json':
        try:
            return json.loads(raw_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif format_hint == 'ini':
        try:
            config = configparser.ConfigParser()
            config.read_string(raw_bytes.decode('utf-8'))
            # Convert ConfigParser to dictionary
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            return result
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid INI data: {e}")
