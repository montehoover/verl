import json
import logging
from html.parser import HTMLParser

# Configure logging
logger = logging.getLogger(__name__)

allowed_formats = ["json", "html"]

def validate_format(stream_format):
    if stream_format not in allowed_formats:
        logger.error(f"Format validation failed: {stream_format} is not in allowed formats {allowed_formats}")
        raise ValueError(f"Unsupported or insecure format: {stream_format}")
    logger.debug(f"Format validation successful: {stream_format}")

def deserialize_json(stream):
    try:
        logger.debug("Attempting JSON deserialization")
        result = json.loads(stream.decode('utf-8'))
        logger.info("JSON deserialization successful")
        return result
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"JSON deserialization failed: {type(e).__name__}: {e}")
        raise ValueError(f"Invalid JSON data: {e}")

def deserialize_html(stream):
    try:
        logger.debug("Attempting HTML deserialization")
        decoded_html = stream.decode('utf-8')
        parser = HTMLParser()
        parser.feed(decoded_html)
        logger.info("HTML deserialization successful")
        return decoded_html
    except UnicodeDecodeError as e:
        logger.error(f"HTML deserialization failed: {type(e).__name__}: {e}")
        raise ValueError(f"Invalid HTML data: {e}")

def deserialize_content_stream(stream, stream_format):
    logger.info(f"Deserialization attempt started for format: {stream_format}")
    
    try:
        validate_format(stream_format)
        
        if stream_format == "json":
            result = deserialize_json(stream)
        elif stream_format == "html":
            result = deserialize_html(stream)
        
        logger.info(f"Deserialization completed successfully for format: {stream_format}")
        return result
    except Exception as e:
        logger.error(f"Deserialization failed for format: {stream_format} - {type(e).__name__}: {e}")
        raise
