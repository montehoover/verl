import json
import html
import logging
from datetime import datetime

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

allowed_formats = ["json", "html"]

def validate_format(stream_format: str) -> None:
    """Validate that the format is allowed and secure."""
    if stream_format not in allowed_formats:
        logger.error(f"Validation failed: Format '{stream_format}' is either unsupported or considered insecure")
        raise ValueError(f"Format '{stream_format}' is either unsupported or considered insecure")

def deserialize_json(serialized_data: bytes) -> object:
    """Deserialize JSON data from bytes."""
    try:
        result = json.loads(serialized_data.decode('utf-8'))
        logger.info(f"Successfully deserialized JSON data (size: {len(serialized_data)} bytes)")
        return result
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"JSON deserialization failed: {str(e)}")
        raise ValueError(f"Failed to deserialize JSON data: {str(e)}")

def deserialize_html(serialized_data: bytes) -> str:
    """Deserialize HTML data from bytes."""
    try:
        decoded_str = serialized_data.decode('utf-8')
        result = html.unescape(decoded_str)
        logger.info(f"Successfully deserialized HTML data (size: {len(serialized_data)} bytes)")
        return result
    except UnicodeDecodeError as e:
        logger.error(f"HTML deserialization failed: {str(e)}")
        raise ValueError(f"Failed to deserialize HTML data: {str(e)}")

def deserialize_stream(serialized_data: bytes, stream_format: str):
    logger.info(f"Deserialization attempt started - Format: '{stream_format}', Data size: {len(serialized_data)} bytes")
    
    try:
        validate_format(stream_format)
        
        deserializers = {
            "json": deserialize_json,
            "html": deserialize_html
        }
        
        result = deserializers[stream_format](serialized_data)
        logger.info(f"Deserialization completed successfully - Format: '{stream_format}'")
        return result
    except Exception as e:
        logger.error(f"Deserialization failed - Format: '{stream_format}', Error: {str(e)}")
        raise
