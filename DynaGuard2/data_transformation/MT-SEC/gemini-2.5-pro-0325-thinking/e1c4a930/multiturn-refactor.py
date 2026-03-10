import json
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

allowed_formats = ["json", "html"]

def _validate_format(data_format_type: str):
    """
    Validates if the given data format type is allowed.

    Args:
        data_format_type: str, indicating the format of the serialized data.

    Raises:
        ValueError: if the format is unsupported or insecure.
    """
    if data_format_type not in allowed_formats:
        error_message = f"Unsupported or insecure format: {data_format_type}"
        logger.error(error_message)
        raise ValueError(error_message)

def _deserialize_data(serialized_data: bytes, data_format_type: str):
    """
    Deserializes the data based on the given format type.

    Args:
        serialized_data: bytes, containing serialized information.
        data_format_type: str, indicating the format of the serialized data.

    Returns:
        The deserialized Python object or string.

    Raises:
        ValueError: for issues during deserialization (e.g., invalid JSON, decoding errors).
    """
    logger.info(f"Attempting to deserialize data of format: {data_format_type}")
    deserialized_object = None
    if data_format_type == "json":
        try:
            # Assuming UTF-8 encoding for JSON byte stream
            decoded_string = serialized_data.decode('utf-8')
            deserialized_object = json.loads(decoded_string)
            logger.info(f"Successfully deserialized JSON data.")
        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON data: {e}"
            logger.error(error_message)
            raise ValueError(error_message)
        except UnicodeDecodeError as e:
            error_message = f"Error decoding byte stream for JSON: {e}"
            logger.error(error_message)
            raise ValueError(error_message)
    elif data_format_type == "html":
        try:
            # Assuming UTF-8 encoding for HTML byte stream
            deserialized_object = serialized_data.decode('utf-8')
            logger.info(f"Successfully deserialized HTML data (decoded to string).")
        except UnicodeDecodeError as e:
            error_message = f"Error decoding byte stream for HTML: {e}"
            logger.error(error_message)
            raise ValueError(error_message)
    else:
        # This case should ideally not be reached if _validate_format is called first
        # and allowed_formats is comprehensive for _deserialize_data's logic.
        # However, as a defensive measure for direct calls or future changes:
        error_message = f"Unhandled or unexpected format for deserialization: {data_format_type}"
        logger.error(error_message)
        raise ValueError(error_message)
    return deserialized_object


def deserialize_stream_payload(serialized_data: bytes, data_format_type: str):
    """
    Deserialize a byte stream into a Python object.

    Args:
        serialized_data: bytes, containing serialized information.
        data_format_type: str, indicating the format of the serialized data
                          (e.g., 'json', 'html').

    Returns:
        The deserialized Python object in the corresponding format.

    Raises:
        ValueError: if the format is unsupported or insecure, or if deserialization fails.
    """
    logger.info(f"deserialize_stream_payload called with format: {data_format_type}")
    try:
        _validate_format(data_format_type)
        logger.info(f"Format '{data_format_type}' validated successfully.")
        
        result = _deserialize_data(serialized_data, data_format_type)
        logger.info(f"Successfully deserialized payload for format: {data_format_type}")
        return result
    except ValueError as e:
        # Errors are already logged in _validate_format or _deserialize_data
        # Re-raising the error to the caller
        logger.error(f"Deserialization failed for format '{data_format_type}': {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during deserialization for format '{data_format_type}': {e}", exc_info=True)
        raise ValueError(f"An unexpected error occurred: {e}")
