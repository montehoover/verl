import json
import logging
import os

# Setup logging
log_file_path = os.path.join(os.getcwd(), "deserialization.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        # logging.StreamHandler() # Uncomment to also log to console
    ]
)
logger = logging.getLogger(__name__)

allowed_formats = ["json", "html"]


def _validate_stream_format(stream_format: str, current_allowed_formats: list[str]):
    """
    Checks if the stream format is allowed.

    Raises:
        ValueError: If the format is not in current_allowed_formats.
    """
    if stream_format not in current_allowed_formats:
        error_message = (
            f"Unsupported or insecure format: {stream_format}. "
            f"Allowed formats are: {current_allowed_formats}"
        )
        logger.error(f"Format validation failed: {error_message}")
        raise ValueError(error_message)

def _perform_actual_deserialization(serialized_data: bytes, stream_format: str):
    """
    Performs the deserialization based on the stream format.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If deserialization fails or the format is unexpectedly unhandled.
    """
    logger.info(f"Attempting to deserialize data with format: {stream_format}")
    try:
        if stream_format == "json":
            try:
                deserialized_object = json.loads(serialized_data.decode('utf-8'))
                logger.info(f"Successfully deserialized JSON data.")
                return deserialized_object
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON data for format '{stream_format}': {e}")
                raise ValueError(f"Invalid JSON data: {e}")
            except UnicodeDecodeError as e:
                logger.error(f"Error decoding JSON data as UTF-8 for format '{stream_format}': {e}")
                raise ValueError(f"Error decoding JSON data as UTF-8: {e}")
        elif stream_format == "html":
            # For HTML, we'll return the decoded string.
            try:
                decoded_string = serialized_data.decode('utf-8')
                logger.info(f"Successfully decoded HTML data to string.")
                return decoded_string
            except UnicodeDecodeError as e:
                logger.error(f"Error decoding HTML data as UTF-8 for format '{stream_format}': {e}")
                raise ValueError(f"Error decoding HTML data as UTF-8: {e}")
        else:
            # This case implies an internal inconsistency.
            error_message = (
                f"Internal error: Format '{stream_format}' passed validation "
                "but has no implemented deserialization logic."
            )
            logger.error(error_message)
            raise ValueError(error_message)
    except ValueError as e:
        logger.error(f"Deserialization failed for format '{stream_format}': {e}")
        raise # Re-raise the caught ValueError to propagate it

def deserialize_stream(serialized_data: bytes, stream_format: str):
    """
    Deserialize a given byte stream back into a Python object.

    Args:
        serialized_data: bytes, the serialized data that needs to be deserialized.
        stream_format: str, a string indicating the deserialization format (e.g., 'json', 'html').

    Returns:
        The corresponding Python object reconstructed from the serialized content.

    Raises:
        ValueError: If the specified format is either unsupported or considered insecure.
    """
    logger.info(f"deserialize_stream called with format: {stream_format}")
    try:
        _validate_stream_format(stream_format, allowed_formats)
        result = _perform_actual_deserialization(serialized_data, stream_format)
        logger.info(f"Deserialization successful for format: {stream_format}")
        return result
    except ValueError as e:
        # The specific error is already logged by the helper functions.
        # Here we log the overall failure of the deserialize_stream operation.
        logger.error(f"deserialize_stream failed for format '{stream_format}'. Error: {e}")
        raise
