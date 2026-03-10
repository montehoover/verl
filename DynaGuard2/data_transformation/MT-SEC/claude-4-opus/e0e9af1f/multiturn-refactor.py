import json
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

acceptable_formats = ["json", "text"]

def validate_format(data_format: str) -> None:
    if data_format not in acceptable_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")

def deserialize_json(file_location: str):
    with open(file_location, 'r') as f:
        return json.load(f)

def deserialize_text(file_location: str):
    with open(file_location, 'r') as f:
        return f.read()

def load_serialized_data(file_location: str, data_format: str):
    logger.info(f"Attempting to deserialize file: {file_location} with format: {data_format}")
    
    try:
        validate_format(data_format)
    except ValueError as e:
        logger.error(f"Format validation failed for file: {file_location}, format: {data_format}, error: {str(e)}")
        raise
    
    try:
        if data_format == "json":
            result = deserialize_json(file_location)
        elif data_format == "text":
            result = deserialize_text(file_location)
        
        logger.info(f"Successfully deserialized file: {file_location} with format: {data_format}")
        return result
    except Exception as e:
        logger.error(f"Deserialization failed for file: {file_location}, format: {data_format}, error: {str(e)}")
        raise
