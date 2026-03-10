import json
import logging

# Configure logging
# For a real application, you might want to configure this in a more centralized way
# (e.g., in your application's entry point)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

safe_formats = ["json", "html"]


def _deserialize_json(byte_data: bytes):
    """Deserializes JSON byte data to a Python object."""
    try:
        # Assuming UTF-8 encoding for the byte stream for JSON
        decoded_string = byte_data.decode('utf-8')
        obj = json.loads(decoded_string)
        logger.info("Successfully deserialized JSON data.")
        return obj
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.error(f"Failed to deserialize JSON data: {e}", exc_info=True)
        raise ValueError(f"Failed to deserialize JSON data: {e}")


def _deserialize_html(byte_data: bytes):
    """Deserializes HTML byte data to a string."""
    try:
        # For HTML, "deserialization" typically means decoding to a string
        # Assuming UTF-8 encoding, common for HTML
        decoded_string = byte_data.decode('utf-8')
        logger.info("Successfully decoded HTML data.")
        return decoded_string
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode HTML data: {e}", exc_info=True)
        raise ValueError(f"Failed to decode HTML data: {e}")


def restore_object_from_stream(byte_data: bytes, serialization_type: str):
    """
    Deserializes a byte stream into its corresponding Python object.

    Only known, secure formats are accepted for deserialization.

    Args:
        byte_data: bytes, the serialized stream of data.
        serialization_type: str, specifies the format of the serialized data
                            (e.g., 'json', 'html').

    Returns:
        The Python object reconstructed from the byte stream.

    Raises:
        ValueError: If deserialization with an unsupported or unsafe format
                    is attempted.
    """
    logger.info(f"Attempting to deserialize data of type: {serialization_type}")
    if serialization_type not in safe_formats:
        error_message = f"Unsupported or unsafe serialization format: {serialization_type}"
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        if serialization_type == "json":
            obj = _deserialize_json(byte_data)
            logger.info(f"Successfully restored object from stream for type: {serialization_type}")
            return obj
        elif serialization_type == "html":
            obj = _deserialize_html(byte_data)
            logger.info(f"Successfully restored object from stream for type: {serialization_type}")
            return obj
        
        # This part should ideally not be reached if safe_formats check is comprehensive
        # and covers all implemented branches.
        # However, as a safeguard:
        # Defensive coding: although the initial check for `serialization_type not in safe_formats`
        # should prevent reaching here with an unknown type, this ensures that if `safe_formats`
        # were to include a type without a corresponding deserialization branch,
        # an error is still raised.
        internal_error_message = f"Internal error: No deserialization logic for safe format '{serialization_type}'"
        logger.error(internal_error_message)
        raise ValueError(internal_error_message)
    except ValueError as e:
        # Log the error that was raised from helper functions or the initial check
        # This ensures that even if helper functions log, the main function also logs the failure context
        logger.error(f"Deserialization failed for type '{serialization_type}': {e}", exc_info=True)
        raise # Re-raise the original exception to maintain existing behavior
