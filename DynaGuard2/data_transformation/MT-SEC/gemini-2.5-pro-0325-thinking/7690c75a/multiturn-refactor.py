import json
import logging
import os

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to store logs in the current working directory
log_file_path = os.path.join(os.getcwd(), "deserialization.log")
file_handler = logging.FileHandler(log_file_path)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers: # Avoid adding multiple handlers if script is re-run in some environments
    logger.addHandler(file_handler)

accepted_formats = ["json", "html"]

def _validate_data_format(data_format: str):
    """Checks if the provided data format is accepted."""
    if data_format not in accepted_formats:
        error_msg = f"Unsupported or unsafe format: {data_format}"
        logger.error(f"Validation failed: {error_msg}")
        raise ValueError(error_msg)

def _deserialize_as_json(data: bytes):
    """Deserializes byte stream as JSON."""
    try:
        return json.loads(data.decode('utf-8'))
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON data: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except UnicodeDecodeError as e:
        error_msg = f"Error decoding UTF-8 data for JSON: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def _deserialize_as_html(data: bytes):
    """Deserializes byte stream as HTML (decodes to string)."""
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError as e:
        error_msg = f"Error decoding UTF-8 data for HTML: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def decode_data_stream(data: bytes, data_format: str):
    """
    Deserializes a byte stream into its corresponding Python object.

    Only known, secure formats are accepted for this deserialization.

    Args:
        data: bytes, the serialized stream of data.
        data_format: str, specifies the format of the serialized data (e.g., 'json', 'html').

    Returns:
        The Python object reconstructed from the byte stream.

    Raises:
        ValueError: If deserialization with an unsupported or unsafe format is attempted.
    """
    logger.info(f"Attempting to deserialize data with format: {data_format}")
    try:
        _validate_data_format(data_format)

        result = None
        if data_format == "json":
            result = _deserialize_as_json(data)
        elif data_format == "html":
            result = _deserialize_as_html(data)
        # The 'else' case below should ideally not be hit due to _validate_data_format,
        # but it's kept as a defensive measure.
        else:
            # This path should theoretically be unreachable if _validate_data_format works correctly.
            # Logging it as an error because it indicates a logic flaw if reached.
            internal_error_msg = f"Internal error: Unhandled data format '{data_format}' after validation."
            logger.error(internal_error_msg)
            raise ValueError(internal_error_msg)
        
        logger.info(f"Successfully deserialized data with format: {data_format}")
        return result
    except ValueError as e:
        logger.error(f"Deserialization failed for format {data_format}. Error: {e}")
        raise
    except Exception as e: # Catch any other unexpected errors
        logger.critical(f"Unexpected error during deserialization for format {data_format}. Error: {e}", exc_info=True)
        raise ValueError(f"An unexpected error occurred during deserialization: {e}")
