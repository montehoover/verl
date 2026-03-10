import json
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler
handler = logging.FileHandler('deserialization.log')
handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

trusted_formats = ["json", "text"]

def validate_format(file_format: str) -> None:
    if file_format not in trusted_formats:
        raise ValueError(f"Unsafe or unsupported format: {file_format}")

def deserialize_content(file_path: str, file_format: str):
    if file_format == "json":
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_format == "text":
        with open(file_path, 'r') as f:
            return f.read()

def read_serialized_file(file_path: str, file_format: str):
    try:
        validate_format(file_format)
        logger.info(f"Attempting to deserialize file: {file_path} with format: {file_format}")
        
        result = deserialize_content(file_path, file_format)
        
        logger.info(f"Successfully deserialized file: {file_path} with format: {file_format}")
        return result
        
    except ValueError as e:
        logger.error(f"Format validation failed for file: {file_path} with format: {file_format} - Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Deserialization failed for file: {file_path} with format: {file_format} - Error: {str(e)}")
        raise
