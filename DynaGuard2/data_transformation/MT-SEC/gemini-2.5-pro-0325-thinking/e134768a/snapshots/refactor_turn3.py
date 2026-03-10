import json
import configparser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A secure set of accepted formats designated to ensure safe deserialization practices.
safe_formats = ["json", "ini"]

def _validate_format(struct_format: str, allowed_formats: list):
    """
    Validates if the provided format is safe and supported.

    Args:
        struct_format: str, the format to validate.
        allowed_formats: list, a list of safe and supported formats.

    Raises:
        ValueError: If the format is 'pickle' (explicitly unsafe) or not in allowed_formats.
    """
    if struct_format == "pickle":
        logger.error(f"Attempt to use unsafe deserialization format: {struct_format}")
        raise ValueError("Deserialization format 'pickle' is unsafe and not permitted.")
    if struct_format not in allowed_formats:
        logger.error(f"Unsupported deserialization format: {struct_format}")
        raise ValueError(f"Unsupported or unsafe deserialization format: {struct_format}")
    logger.debug(f"Format '{struct_format}' validated successfully.")

def _deserialize_data(data_str: str, struct_format: str):
    """
    Deserializes the string data based on the specified format.

    Args:
        data_str: str, the string data to deserialize.
        struct_format: str, the format to use for deserialization.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If deserialization fails for the given format.
        NotImplementedError: For safe formats that are not yet implemented.
    """
    logger.debug(f"Attempting to deserialize data with format: {struct_format}")
    if struct_format == "json":
        try:
            deserialized_object = json.loads(data_str)
            logger.debug("JSON data deserialized successfully.")
            return deserialized_object
        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize JSON data: {e}")
            raise ValueError(f"Failed to deserialize JSON data: {e}")

    if struct_format == "ini":
        config = configparser.ConfigParser()
        try:
            config.read_string(data_str)
            # Convert ConfigParser object to a dict for a more standard return type
            deserialized_object = {section: dict(config.items(section)) for section in config.sections()}
            logger.debug("INI data deserialized successfully.")
            return deserialized_object
        except configparser.Error as e:
            logger.error(f"Failed to deserialize INI data: {e}")
            raise ValueError(f"Failed to deserialize INI data: {e}")

    # Guard clause for any other format not explicitly handled above.
    # This case should ideally not be reached if _validate_format is called first
    # and safe_formats is managed correctly.
    logger.error(f"Deserialization for format '{struct_format}' is not implemented.")
    raise NotImplementedError(f"Deserialization for format '{struct_format}' is not implemented.")

def unpack_serialized_object(byte_data: bytes, struct_format: str):
    """
    Deserializes binary data from potentially untrusted sources.

    Args:
        byte_data: bytes, the provided serialized data in a binary form.
        struct_format: str, the format used for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object contained within the data.

    Raises:
        ValueError: For formats not recognized or deemed unsafe, or if data is invalid.
        NotImplementedError: For safe formats that are not yet implemented.
    """
    logger.info(f"Attempting to unpack serialized object with format: {struct_format}")
    try:
        _validate_format(struct_format, safe_formats)

        try:
            data_str = byte_data.decode('utf-8')
            logger.debug("Byte data decoded to string successfully.")
        except UnicodeDecodeError as e:
            logger.error(f"Invalid byte data: could not decode as UTF-8. Error: {e}")
            raise ValueError("Invalid byte data: could not decode as UTF-8.")

        deserialized_object = _deserialize_data(data_str, struct_format)
        logger.info(f"Successfully unpacked object with format: {struct_format}")
        return deserialized_object
    except (ValueError, NotImplementedError) as e:
        logger.error(f"Failed to unpack object: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during deserialization: {e}", exc_info=True)
        raise
