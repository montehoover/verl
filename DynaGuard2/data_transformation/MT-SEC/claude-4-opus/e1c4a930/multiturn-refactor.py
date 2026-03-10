import json
import html
import logging

# Configure logger
logger = logging.getLogger(__name__)

allowed_formats = ["json", "html"]

def validate_format(data_format_type: str) -> None:
    if data_format_type not in allowed_formats:
        logger.error(f"Validation failed: Unsupported or insecure format: {data_format_type}")
        raise ValueError(f"Unsupported or insecure format: {data_format_type}")

def deserialize_json(serialized_data: bytes):
    return json.loads(serialized_data.decode('utf-8'))

def deserialize_html(serialized_data: bytes):
    return html.unescape(serialized_data.decode('utf-8'))

def deserialize_stream_payload(serialized_data: bytes, data_format_type: str):
    logger.info(f"Deserialization attempt started for format: {data_format_type}")
    
    try:
        validate_format(data_format_type)
        
        if data_format_type == "json":
            result = deserialize_json(serialized_data)
        elif data_format_type == "html":
            result = deserialize_html(serialized_data)
        
        logger.info(f"Deserialization successful for format: {data_format_type}")
        return result
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON deserialization failed: {str(e)}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error during deserialization: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Deserialization failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during deserialization: {str(e)}")
        raise
