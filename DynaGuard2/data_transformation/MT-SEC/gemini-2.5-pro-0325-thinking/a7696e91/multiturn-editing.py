import json
import configparser
import logging
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a constant for the size limit (e.g., 1MB)
MAX_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB

# Define approved serialization formats
APPROVED_FORMATS = ["json", "ini"]

class SizeLimitExceededError(ValueError):
    """Custom exception for when input data exceeds the size limit."""
    pass

class UnsafeFormatError(ValueError):
    """Custom exception for when an unsupported or unsafe serialization format is provided."""
    pass

class ParsingError(ValueError):
    """Custom exception for errors during parsing."""
    pass

def load_serialized_content(serialized_input: bytes, serialization_format: str) -> Any:
    """
    Reads binary data, checks its size, and attempts to parse it if it's in an approved format.

    Args:
        serialized_input: A bytes object containing the serialized data.
        serialization_format: The expected format of the data (must be in APPROVED_FORMATS).

    Returns:
        Parsed data (dict for JSON, ConfigParser object for INI).

    Raises:
        SizeLimitExceededError: If the binary data is too large.
        TypeError: If serialized_input is not bytes or serialization_format is not str.
        UnsafeFormatError: If the serialization_format is not in APPROVED_FORMATS.
        ParsingError: If parsing fails for an approved format.
        UnicodeDecodeError: If the binary data cannot be decoded to UTF-8.
    """
    if not isinstance(serialized_input, bytes):
        raise TypeError("Input 'serialized_input' must be a bytes object.")
    if not isinstance(serialization_format, str):
        raise TypeError("Input 'serialization_format' must be a string.")

    format_lower = serialization_format.lower()
    logger.info(f"Received data of size {len(serialized_input)} bytes, requested format: {format_lower}")

    if format_lower not in APPROVED_FORMATS:
        logger.error(f"Unsupported or unsafe serialization format requested: '{serialization_format}'. Approved formats: {APPROVED_FORMATS}")
        raise UnsafeFormatError(
            f"Format '{serialization_format}' is not approved. "
            f"Approved formats are: {', '.join(APPROVED_FORMATS)}."
        )

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
        # Re-raise with a more specific message or handle as needed.
        # This is a critical step for text-based formats.
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, "Failed to decode input as UTF-8.")


    if format_lower == "json":
        logger.info("Attempting to parse content as JSON.")
        try:
            parsed_data = json.loads(decoded_content)
            logger.info("Successfully parsed content as JSON.")
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            raise ParsingError(f"Failed to parse JSON content: {e}")
        except Exception as e: # Catch any other unexpected errors during JSON processing
            logger.error(f"An unexpected error occurred during JSON parsing: {e}")
            raise ParsingError(f"An unexpected error occurred during JSON parsing: {e}")

    elif format_lower == "ini":
        logger.info("Attempting to parse content as INI.")
        try:
            config = configparser.ConfigParser()
            # configparser reads directly from string, no need to handle file-like objects for this basic case
            config.read_string(decoded_content)
            logger.info("Successfully parsed content as INI.")
            # To make it behave more like json.loads, which returns a dict-like structure,
            # we can convert the ConfigParser object to a dictionary of dictionaries.
            # This is an opinionated choice for consistency, adjust if direct ConfigParser object is preferred.
            ini_data = {section: dict(config.items(section)) for section in config.sections()}
            return ini_data
        except configparser.Error as e: # Catches MissingSectionHeaderError, ParsingError, etc.
            logger.error(f"INI parsing failed: {e}")
            raise ParsingError(f"Failed to parse INI content: {e}")
        except Exception as e: # Catch any other unexpected errors during INI processing
            logger.error(f"An unexpected error occurred during INI parsing: {e}")
            raise ParsingError(f"An unexpected error occurred during INI parsing: {e}")

    # This part should ideally not be reached if format_lower is in APPROVED_FORMATS
    # and all approved formats are handled above.
    # However, as a safeguard:
    logger.error(f"Internal error: Approved format '{format_lower}' not handled.")
    raise ParsingError(f"Internal error: No parser implemented for approved format '{format_lower}'.")
