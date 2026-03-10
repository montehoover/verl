import json
import configparser
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deserialization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

valid_formats = ["json", "ini"]

def validate_format(format_hint: str) -> None:
    """Validate that the format is supported and safe."""
    if format_hint not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint}")

def deserialize_json(raw_bytes: bytes) -> object:
    """Deserialize JSON data from bytes."""
    try:
        return json.loads(raw_bytes.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to deserialize JSON data: {e}")

def deserialize_ini(raw_bytes: bytes) -> dict:
    """Deserialize INI data from bytes."""
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

def convert_serialized_data(raw_bytes: bytes, format_hint: str):
    logger.info(f"Attempting deserialization with format: {format_hint}")
    
    try:
        validate_format(format_hint)
        
        if format_hint == "json":
            result = deserialize_json(raw_bytes)
        elif format_hint == "ini":
            result = deserialize_ini(raw_bytes)
        
        logger.info(f"Successfully deserialized data using format: {format_hint}")
        return result
    
    except Exception as e:
        logger.error(f"Deserialization failed for format {format_hint}: {type(e).__name__}: {str(e)}")
        raise
