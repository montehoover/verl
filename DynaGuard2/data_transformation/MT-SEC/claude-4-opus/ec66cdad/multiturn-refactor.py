import json
import html
import logging

# Configure logger
logger = logging.getLogger(__name__)

safe_formats = ["json", "html"]

def deserialize_json(byte_data: bytes) -> object:
    return json.loads(byte_data.decode('utf-8'))

def deserialize_html(byte_data: bytes) -> str:
    return html.unescape(byte_data.decode('utf-8'))

def restore_object_from_stream(byte_data: bytes, serialization_type: str):
    logger.info(f"Attempting deserialization with format: {serialization_type}")
    
    if serialization_type not in safe_formats:
        error_msg = f"Unsupported or unsafe format: {serialization_type}"
        logger.error(f"Deserialization failed: {error_msg}")
        raise ValueError(error_msg)
    
    deserializers = {
        "json": deserialize_json,
        "html": deserialize_html
    }
    
    try:
        result = deserializers[serialization_type](byte_data)
        logger.info(f"Successfully deserialized data using format: {serialization_type}")
        return result
    except Exception as e:
        error_msg = f"Deserialization failed for format {serialization_type}: {str(e)}"
        logger.error(error_msg)
        raise
