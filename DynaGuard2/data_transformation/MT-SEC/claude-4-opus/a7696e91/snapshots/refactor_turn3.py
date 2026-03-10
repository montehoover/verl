import json
import configparser
from io import StringIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

approved_formats = ["json", "ini"]

def load_serialized_content(serialized_input: bytes, serialization_format: str):
    logger.info(f"Deserialization attempt started for format: '{serialization_format}'")
    
    if serialization_format not in approved_formats:
        error_msg = f"Format '{serialization_format}' is not recognized or is unsafe"
        logger.error(f"Deserialization failed: {error_msg}")
        raise ValueError(error_msg)
    
    try:
        if serialization_format == "json":
            result = json.loads(serialized_input.decode('utf-8'))
            logger.info(f"Successfully deserialized JSON data")
            return result
        
        if serialization_format == "ini":
            config = configparser.ConfigParser()
            config.read_string(serialized_input.decode('utf-8'))
            # Convert ConfigParser to dictionary
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            logger.info(f"Successfully deserialized INI data with {len(result)} section(s)")
            return result
            
    except Exception as e:
        logger.error(f"Deserialization failed for format '{serialization_format}': {type(e).__name__}: {str(e)}")
        raise
