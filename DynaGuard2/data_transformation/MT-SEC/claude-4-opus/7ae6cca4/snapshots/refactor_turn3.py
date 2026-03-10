import json
import configparser
import io
import logging
import os

# Configure logging to write to current working directory
log_file = os.path.join(os.getcwd(), 'deserialization.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

trusted_formats = ["json", "ini"]

def _deserialize_json(data_bytes):
    """Deserialize JSON data from bytes."""
    try:
        logger.info("Starting JSON deserialization")
        result = json.loads(data_bytes.decode('utf-8'))
        logger.info("JSON deserialization successful")
        return result
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"JSON deserialization failed: {e}")
        raise ValueError(f"Failed to decode JSON data: {e}")

def _deserialize_ini(data_bytes):
    """Deserialize INI data from bytes."""
    try:
        logger.info("Starting INI deserialization")
        config = configparser.ConfigParser()
        config.read_string(data_bytes.decode('utf-8'))
        # Convert ConfigParser to dict
        result = {}
        for section in config.sections():
            result[section] = dict(config.items(section))
        logger.info("INI deserialization successful")
        return result
    except (configparser.Error, UnicodeDecodeError) as e:
        logger.error(f"INI deserialization failed: {e}")
        raise ValueError(f"Failed to decode INI data: {e}")

def decode_serialized_data(data_bytes, format_string):
    logger.info(f"Deserialization requested for format: {format_string}")
    
    if format_string not in trusted_formats:
        logger.warning(f"Attempted to use untrusted format: {format_string}")
        raise ValueError(f"Untrusted format: {format_string}")
    
    deserializers = {
        "json": _deserialize_json,
        "ini": _deserialize_ini
    }
    
    deserializer = deserializers.get(format_string)
    if deserializer is None:
        logger.error(f"Unsupported format requested: {format_string}")
        raise ValueError(f"Unsupported format: {format_string}")
    
    try:
        result = deserializer(data_bytes)
        logger.info(f"Successfully deserialized data using format: {format_string}")
        return result
    except Exception as e:
        logger.error(f"Deserialization failed for format {format_string}: {e}")
        raise
