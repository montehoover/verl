import json
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

allowed_formats = ["json", "html"]

def _validate_format(stream_format: str):
    """Checks if the stream_format is in the globally defined allowed_formats."""
    if stream_format not in allowed_formats:
        logger.error(f"Validation failed: Unsupported or insecure format attempted: {stream_format}")
        raise ValueError(f"Unsupported or insecure format: {stream_format}")

def _deserialize_json(stream: bytes):
    """Deserializes a byte stream assumed to be JSON."""
    logger.info("Attempting to deserialize JSON stream.")
    try:
        # Decode bytes to string before parsing JSON
        decoded_stream = stream.decode('utf-8')
        data = json.loads(decoded_stream)
        logger.info("Successfully deserialized JSON stream.")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON deserialization failed: Invalid JSON data. Error: {e}")
        raise ValueError(f"Invalid JSON data: {e}")
    except UnicodeDecodeError as e:
        logger.error(f"JSON deserialization failed: Error decoding stream. Error: {e}")
        raise ValueError(f"Error decoding stream for JSON: {e}")

def _deserialize_html(stream: bytes):
    """Deserializes a byte stream assumed to be HTML (decodes to string)."""
    logger.info("Attempting to deserialize HTML stream.")
    try:
        # For HTML, we typically just decode to a string
        decoded_stream = stream.decode('utf-8')
        logger.info("Successfully deserialized HTML stream.")
        return decoded_stream
    except UnicodeDecodeError as e:
        logger.error(f"HTML deserialization failed: Error decoding stream. Error: {e}")
        raise ValueError(f"Error decoding stream for HTML: {e}")

def deserialize_content_stream(stream: bytes, stream_format: str):
    """
    Deserialize a byte stream into a Python object.

    Args:
        stream: bytes, containing serialized information.
        stream_format: str, indicating the format of the serialized data
                       (e.g., 'json', 'html').

    Returns:
        The deserialized Python object in the corresponding format.

    Raises:
        ValueError: If the format is unsupported or insecure.
    """
    logger.info(f"Attempting to deserialize content with format: {stream_format}")
    try:
        _validate_format(stream_format)

        if stream_format == "json":
            result = _deserialize_json(stream)
        elif stream_format == "html":
            result = _deserialize_html(stream)
        else:
            # This case should ideally not be reached if _validate_format ensures
            # stream_format is one of the allowed_formats, and all allowed_formats
            # have a corresponding handler in this function.
            # This signifies an internal inconsistency.
            err_msg = f"Internal error: Format '{stream_format}' is allowed but not explicitly handled."
            logger.critical(err_msg)
            raise RuntimeError(err_msg)
        
        logger.info(f"Successfully deserialized content with format: {stream_format}")
        return result
    except ValueError as e:
        logger.error(f"Failed to deserialize content with format {stream_format}: {e}")
        raise
    except RuntimeError as e:
        # This catches the specific RuntimeError from the 'else' block above.
        # No need to re-log critically, as it's already done.
        raise
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.exception(f"An unexpected error occurred during deserialization for format {stream_format}: {e}")
        raise ValueError(f"An unexpected error occurred during deserialization: {e}")
