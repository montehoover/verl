import json
import configparser
import pickle
import logging
from io import StringIO

# Configure logging
logger = logging.getLogger(__name__)

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
        logger.warning(f"Attempted to use unsafe format: {struct_format}")
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
        result = json.loads(byte_data.decode('utf-8'))
        logger.debug(f"Successfully deserialized JSON data")
        return result
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Failed to deserialize JSON data: {e}")
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
        logger.debug(f"Successfully deserialized INI data with {len(result)} sections")
        return result
    except (configparser.Error, UnicodeDecodeError) as e:
        logger.error(f"Failed to deserialize INI data: {e}")
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
    logger.info(f"Attempting to deserialize data with format: {struct_format}")
    
    # Guard clause for format safety
    check_format_safety(struct_format)
    
    # Guard clause for JSON format
    if struct_format == "json":
        return deserialize_json(byte_data)
    
    # Guard clause for INI format
    if struct_format == "ini":
        return deserialize_ini(byte_data)
    
    # This should never be reached due to check_format_safety, but included for completeness
    logger.error(f"Unexpected format after safety check: {struct_format}")
    raise ValueError(f"Unexpected format: {struct_format}")
