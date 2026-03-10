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

acceptable_formats = ["json", "text"]

def _validate_data_format(data_format: str, allowed_formats: list[str]):
    """
    Validates if the provided data_format is in the list of allowed formats.

    Args:
        data_format: The format to validate.
        allowed_formats: A list of acceptable format strings.

    Raises:
        ValueError: If the data_format is not in allowed_formats.
    """
    if data_format not in allowed_formats:
        error_message = (
            f"Unsupported or unsafe data format: {data_format}. "
            f"Allowed formats are: {', '.join(allowed_formats)}"
        )
        logger.error(f"Validation Error: {error_message}")
        raise ValueError(error_message)

def _deserialize_file_content(file_handle, data_format: str, file_location: str):
    """
    Deserializes content from an open file handle based on the data_format.

    Args:
        file_handle: An open file object to read from.
        data_format: The format of the data in the file ('json' or 'text').
        file_location: The path to the file, for error reporting context.

    Returns:
        The deserialized data.

    Raises:
        ValueError: If JSON decoding fails.
    """
    try:
        if data_format == "json":
            return json.load(file_handle)
        elif data_format == "text":
            return file_handle.read()
        # Should not happen if _validate_data_format is called first
        # but as a safeguard for direct calls or future formats:
        else: # pragma: no cover
             raise ValueError(f"Internal error: Unexpected data format '{data_format}' for deserialization.") # pragma: no cover
    except json.JSONDecodeError as e:
        error_message = f"Error decoding JSON from {file_location}: {e}"
        logger.error(f"Deserialization Error: {error_message}")
        raise ValueError(error_message)


def load_serialized_data(file_location: str, data_format: str):
    """
    Deserializes data from an external file, supporting only secure formats.

    Args:
        file_location: str, path to the file containing the serialized data.
        data_format: str, indicates the format of the serialized data,
                     restricted to trusted options (e.g. 'json', 'text').

    Returns:
        A Python object that results from deserializing the file contents.

    Raises:
        ValueError: If the format is unsafe or unsupported, or if JSON decoding fails.
        FileNotFoundError: If the file_location does not exist.
        IOError: If there is an issue reading the file.
    """
    logger.info(f"Attempting to load data from '{file_location}' with format '{data_format}'.")
    try:
        _validate_data_format(data_format, acceptable_formats) # ValueError logged inside
        with open(file_location, 'r') as f:
            data = _deserialize_file_content(f, data_format, file_location) # ValueError logged inside
        logger.info(f"Successfully deserialized data from '{file_location}'.")
        return data
    except FileNotFoundError:
        error_message = f"Error: The file was not found at {file_location}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)
    except ValueError as e: # Catches errors from validation or deserialization if not already logged
        # This ensures that if an error was raised before logging inside helpers, it's caught here.
        # Or if a new ValueError type is introduced.
        if not str(e).startswith("Unsupported or unsafe data format") and not str(e).startswith("Error decoding JSON"):
             logger.error(f"ValueError during processing of '{file_location}': {e}")
        raise # Re-raise the error after logging (or if already logged)
    except IOError as e:
        error_message = f"IOError reading file {file_location}: {e}"
        logger.error(error_message)
        raise IOError(error_message)
    except Exception as e: # Catch-all for any other unexpected errors
        error_message = f"An unexpected error occurred while processing '{file_location}': {e}"
        logger.critical(error_message, exc_info=True) # exc_info=True will log stack trace
        raise
