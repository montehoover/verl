import json
from html.parser import HTMLParser
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

allowed_formats = ["json", "html"]

def validate_format(format_type: str) -> None:
    if format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}")

def deserialize_json(content: bytes):
    try:
        return json.loads(content.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}")

def deserialize_html(content: bytes):
    try:
        decoded_content = content.decode('utf-8')
        parser = HTMLParser()
        parser.feed(decoded_content)
        return decoded_content
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode HTML: {str(e)}")

def parse_serialized_content(content: bytes, format_type: str):
    logger.info(f"Deserialization attempt started - Format: {format_type}")
    
    try:
        validate_format(format_type)
        
        if format_type == "json":
            result = deserialize_json(content)
            logger.info(f"Successful deserialization - Format: {format_type}")
            return result
        elif format_type == "html":
            result = deserialize_html(content)
            logger.info(f"Successful deserialization - Format: {format_type}")
            return result
    except ValueError as e:
        logger.error(f"Deserialization failed - Format: {format_type}, Error: {str(e)}")
        raise
