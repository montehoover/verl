import json
import configparser
import logging
import os

# Setup logging
log_file_path = os.path.join(os.getcwd(), "deserialization.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Optionally, to also log to console
    ]
)
logger = logging.getLogger(__name__)

trusted_formats = ["json", "ini"]

def _deserialize_json(data_str: str):
    """Deserializes a JSON string."""
    logger.info("Attempting to deserialize JSON data.")
    try:
        result = json.loads(data_str)
        logger.info("JSON data deserialized successfully.")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e.msg} at pos {e.pos}")
        # Preserve original error information for better debugging
        raise json.JSONDecodeError(f"JSON decoding error: {e.msg}", e.doc, e.pos)

def _deserialize_ini(data_str: str):
    """Deserializes an INI string."""
    logger.info("Attempting to deserialize INI data.")
    try:
        config = configparser.ConfigParser()
        config.read_string(data_str)
        result = {section: dict(config.items(section)) for section in config.sections()}
        logger.info("INI data deserialized successfully.")
        return result
    except configparser.Error as e:
        logger.error(f"INI parsing failed: {e}")
        # Wrap configparser.Error in a ValueError for consistency or re-raise as is
        raise ValueError(f"INI parsing error: {e}")


def decode_serialized_data(data_bytes: bytes, format_string: str):
    """
    Safely deserializes data based on defined trusted formats.

    Args:
        data_bytes: bytes, the incoming serialized data in binary format.
        format_string: str, a string identifying the format of serialization (e.g. 'json', 'ini').

    Returns:
        A Python object resulted from deserializing the data.

    Raises:
        ValueError: For unsupported or untrusted formats.
        json.JSONDecodeError: If JSON decoding fails.
        configparser.Error: If INI parsing fails.
    """
    logger.info(f"decode_serialized_data called with format_string: '{format_string}'")

    if format_string not in trusted_formats:
        logger.error(f"Unsupported or untrusted format specified: {format_string}")
        raise ValueError(f"Unsupported or untrusted format: {format_string}")

    # Pickle is explicitly disallowed for security reasons, even if it were in trusted_formats.
    if format_string == "pickle":
        logger.error("Attempt to use disallowed 'pickle' format.")
        raise ValueError("The 'pickle' format is not allowed due to security concerns.")

    try:
        logger.debug("Attempting to decode data_bytes to UTF-8 string.")
        data_str = data_bytes.decode('utf-8')
        logger.debug("data_bytes decoded to string successfully.")
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode data_bytes to string: {e}")
        raise ValueError(f"Failed to decode data_bytes to string: {e}")


    if format_string == "json":
        logger.info(f"Processing format: {format_string}")
        return _deserialize_json(data_str)
    elif format_string == "ini":
        logger.info(f"Processing format: {format_string}")
        return _deserialize_ini(data_str)
    else:
        # This case should ideally not be reached if trusted_formats check is comprehensive
        # and aligned with the implemented handlers.
        err_msg = f"Internal error: No handler for trusted format: {format_string}"
        logger.error(err_msg)
        raise ValueError(err_msg)
