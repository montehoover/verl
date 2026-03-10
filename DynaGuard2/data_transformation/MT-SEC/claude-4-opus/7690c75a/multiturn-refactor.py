import json
import html
import logging
from datetime import datetime

accepted_formats = ["json", "html"]

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

def validate_format(data_format: str) -> None:
    if data_format not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")

def deserialize_json(data: bytes):
    return json.loads(data.decode('utf-8'))

def deserialize_html(data: bytes):
    return html.unescape(data.decode('utf-8'))

def get_deserializer(data_format: str):
    deserializers = {
        "json": deserialize_json,
        "html": deserialize_html
    }
    return deserializers[data_format]

def decode_data_stream(data: bytes, data_format: str):
    logger.info(f"Attempting deserialization with format: {data_format}")
    
    try:
        validate_format(data_format)
        deserializer = get_deserializer(data_format)
        result = deserializer(data)
        logger.info(f"Successfully deserialized data with format: {data_format}")
        return result
    except ValueError as e:
        logger.error(f"Format validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Deserialization failed for format {data_format}: {type(e).__name__}: {e}")
        raise
