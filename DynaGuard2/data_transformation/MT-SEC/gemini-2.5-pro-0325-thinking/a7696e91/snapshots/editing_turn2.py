import json
import xml.etree.ElementTree as ET
import logging
from typing import Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a constant for the size limit (e.g., 1MB)
MAX_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB
DEFAULT_ERROR_MESSAGE = "Error: Could not parse content."

class SizeLimitExceededError(ValueError):
    """Custom exception for when input data exceeds the size limit."""
    pass

def load_serialized_content(serialized_input: bytes, serialization_format: str) -> Any:
    """
    Reads binary data, checks its size, and attempts to parse it based on the specified format.

    Args:
        serialized_input: A bytes object containing the serialized data.
        serialization_format: The expected format of the data (e.g., "json", "xml", "text").

    Returns:
        Parsed data (dict for JSON, ElementTree for XML, str for text)
        or a default error message string if parsing fails.

    Raises:
        SizeLimitExceededError: If the binary data is too large.
        TypeError: If serialized_input is not bytes or serialization_format is not str.
        UnicodeDecodeError: If the binary data cannot be decoded to UTF-8 for text-based formats.
    """
    if not isinstance(serialized_input, bytes):
        raise TypeError("Input 'serialized_input' must be a bytes object.")
    if not isinstance(serialization_format, str):
        raise TypeError("Input 'serialization_format' must be a string.")

    logger.info(f"Received data of size {len(serialized_input)} bytes, format: {serialization_format}")

    if len(serialized_input) > MAX_SIZE_BYTES:
        logger.error(f"Data size ({len(serialized_input)} bytes) exceeds limit of {MAX_SIZE_BYTES} bytes.")
        raise SizeLimitExceededError(
            f"Input data size ({len(serialized_input)} bytes) "
            f"exceeds the limit of {MAX_SIZE_BYTES} bytes."
        )

    try:
        # First, decode to string as JSON and XML are text-based
        decoded_content = serialized_input.decode('utf-8')
        logger.info("Successfully decoded content to UTF-8 string.")
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode input as UTF-8: {e}")
        # Re-raise with a more specific message or handle as needed
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, "Failed to decode input as UTF-8.")

    format_lower = serialization_format.lower()

    if format_lower == "json":
        logger.info("Attempting to parse content as JSON.")
        try:
            parsed_data = json.loads(decoded_content)
            logger.info("Successfully parsed content as JSON.")
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return DEFAULT_ERROR_MESSAGE
        except Exception as e:
            logger.error(f"An unexpected error occurred during JSON parsing: {e}")
            return DEFAULT_ERROR_MESSAGE

    elif format_lower == "xml":
        logger.info("Attempting to parse content as XML.")
        try:
            parsed_data = ET.fromstring(decoded_content)
            logger.info("Successfully parsed content as XML.")
            return parsed_data
        except ET.ParseError as e:
            logger.error(f"XML parsing failed: {e}")
            return DEFAULT_ERROR_MESSAGE
        except Exception as e:
            logger.error(f"An unexpected error occurred during XML parsing: {e}")
            return DEFAULT_ERROR_MESSAGE

    elif format_lower == "text":
        logger.info("Returning content as plain text.")
        return decoded_content

    else:
        logger.warning(
            f"Unknown or unsupported serialization format: '{serialization_format}'. "
            "Returning content as plain text."
        )
        return decoded_content
