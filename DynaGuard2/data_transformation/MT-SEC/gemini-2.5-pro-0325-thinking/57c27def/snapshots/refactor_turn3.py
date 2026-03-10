import json
import configparser
import logging

# A curated list of formats that have been evaluated to ensure safe deserialization
valid_formats = ["json", "ini"]

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Ensure the handler is not added multiple times if the module is reloaded
if not logger.handlers:
    file_handler = logging.FileHandler("deserialization.log") # Stored in current working directory
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _validate_format(format_hint: str):
    """
    Validates if the provided format_hint is supported.

    Args:
        format_hint: str, the format to validate.

    Raises:
        ValueError: If the format is not in valid_formats.
    """
    if format_hint not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint}")


def _deserialize_json(data_str: str):
    """
    Deserializes a JSON string.

    Args:
        data_str: str, the JSON string to deserialize.

    Returns:
        The Python object after performing JSON deserialization.

    Raises:
        ValueError: If JSON deserialization fails.
    """
    try:
        return json.loads(data_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to deserialize JSON data: {e}")


def _deserialize_ini(data_str: str):
    """
    Deserializes an INI string.

    Args:
        data_str: str, the INI string to deserialize.

    Returns:
        A dictionary representation of the INI data.

    Raises:
        ValueError: If INI deserialization fails.
    """
    config = configparser.ConfigParser()
    try:
        config.read_string(data_str)
        return {section: dict(config.items(section)) for section in config.sections()}
    except configparser.Error as e:
        raise ValueError(f"Failed to deserialize INI data: {e}")


def convert_serialized_data(raw_bytes: bytes, format_hint: str):
    """
    Deserializes binary data from potentially unreliable sources.

    Args:
        raw_bytes: bytes, representing the serialized data in binary form.
        format_hint: str, describing the format to use for deserializing 
                     such as 'json', or 'ini'.

    Returns:
        The Python object after performing the deserialization.

    Raises:
        ValueError: If the format is either unsupported or unsafe.
    """
    logger.info(f"Attempting deserialization for format_hint: '{format_hint}'.")
    try:
        _validate_format(format_hint) # Can raise ValueError

        # It's good practice to decode bytes to string before parsing for text-based formats
        # Assuming UTF-8 encoding, which is common. This might need to be configurable
        # or determined from metadata if available.
        try:
            data_str = raw_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            # Log the specific decoding error here before re-raising
            logger.error(
                f"Failed to decode raw_bytes as UTF-8 for format '{format_hint}'. Original error: {e}",
                exc_info=True
            )
            raise ValueError(f"Failed to decode raw_bytes as UTF-8 for format {format_hint}: {e}") from e

        result = None
        if format_hint == "json":
            result = _deserialize_json(data_str) # Can raise ValueError
        elif format_hint == "ini":
            result = _deserialize_ini(data_str) # Can raise ValueError
        else:
            # This case implies valid_formats contains an entry not handled by the if/elif chain.
            # This is an internal logic error if _validate_format passed.
            # The ValueError raised here will be caught and logged by the handler below.
            # This specific situation should ideally not occur if valid_formats and this
            # function's dispatch logic are kept in sync.
            raise ValueError(f"Internal error: Format '{format_hint}' passed validation but has no deserialization handler.")

        logger.info(f"Successfully deserialized data for format_hint: '{format_hint}'.")
        return result

    except ValueError as e:
        # Catches ValueErrors from _validate_format, decode re-raise, _deserialize_*, or the internal error else-clause.
        logger.error(
            f"Deserialization process failed for format_hint: '{format_hint}'. Error: {e}",
            exc_info=True
        )
        raise
    except Exception as e: # Catch any other unexpected errors not already covered by ValueError
        logger.critical(
            f"An unexpected critical error occurred during deserialization for format_hint: '{format_hint}'. Error: {e}",
            exc_info=True
        )
        raise
