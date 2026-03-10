import json
import configparser
from io import StringIO
import logging
from datetime import datetime

trusted_formats = ["json", "ini"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deserialization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Deserialization attempt started - Format: {serialization_format}")
    
    try:
        validate_format(serialization_format)
        
        if serialization_format == "json":
            result = deserialize_json(byte_data)
            logger.info(f"Deserialization successful - Format: {serialization_format}")
            return result
        elif serialization_format == "ini":
            result = deserialize_ini(byte_data)
            logger.info(f"Deserialization successful - Format: {serialization_format}")
            return result
        else:
            raise NotImplementedError(f"Format {serialization_format} is not implemented")
    
    except Exception as e:
        logger.error(f"Deserialization failed - Format: {serialization_format}, Error: {type(e).__name__}: {str(e)}")
        raise
