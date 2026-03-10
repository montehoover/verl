import json
import xml.etree.ElementTree as ET
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

secure_formats = ["json", "xml"]


def _validate_data_format(data_format: str):
    """
    Validates if the provided data format is secure and supported.

    Args:
        data_format: str, indicating the data format.

    Raises:
        ValueError: If the data format is not in the list of secure_formats.
    """
    if data_format not in secure_formats:
        error_message = (
            f"Unsupported or unsafe data format: {data_format}. "
            f"Allowed formats are: {', '.join(secure_formats)}"
        )
        logger.error(f"Validation failed: {error_message}")
        raise ValueError(error_message)
    logger.info(f"Data format '{data_format}' validated successfully.")


def _deserialize_data(decoded_data: str, data_format: str):
    """
    Deserializes the string data based on the given format.

    Args:
        decoded_data: str, the string representation of the data.
        data_format: str, the format to use for deserialization ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If deserialization logic for a secure format is not implemented.
        json.JSONDecodeError: If JSON decoding fails.
        ET.ParseError: If XML parsing fails.
    """
    try:
        if data_format == "json":
            result = json.loads(decoded_data)
            logger.info(f"Successfully deserialized JSON data.")
            return result
        elif data_format == "xml":
            result = ET.fromstring(decoded_data)
            logger.info(f"Successfully deserialized XML data.")
            return result
        # This part should ideally not be reached if _validate_data_format is called first
        # and secure_formats only contains formats implemented here.
        else:
            # This case implies a format was deemed secure but lacks deserialization logic.
            error_message = f"Deserialization logic for {data_format} not implemented, though listed as secure."
            logger.error(error_message)
            raise ValueError(error_message)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to deserialize JSON data: {e}")
        raise
    except ET.ParseError as e:
        logger.error(f"Failed to deserialize XML data: {e}")
        raise
    except Exception as e: # Catch any other unexpected errors during deserialization
        logger.error(f"An unexpected error occurred during deserialization for format {data_format}: {e}")
        raise


def load_serialized_data(bytes_data: bytes, data_format: str):
    """
    Safely converts serialized data, provided as raw bytes, into its
    corresponding Python object.

    Deserialization is restricted to a predefined list of secure data formats
    since the input data may come from untrusted entities.

    Args:
        bytes_data: bytes, representing the serialized form of the object.
        data_format: str, indicating the data format used for serialization
                     (e.g., 'json', 'xml').

    Returns:
        Deserializes the object and returns the resulting Python data structure
        if the format is valid.

    Raises:
        ValueError: When it encounters an unsupported or unsafe format.
    """
    logger.info(f"Attempting to deserialize data with format: {data_format}")
    try:
        _validate_data_format(data_format)
        
        try:
            decoded_data = bytes_data.decode('utf-8')
            logger.info("Successfully decoded bytes_data to string.")
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode bytes_data to UTF-8 string: {e}")
            raise ValueError(f"Invalid byte sequence for UTF-8 decoding: {e}")

        result = _deserialize_data(decoded_data, data_format)
        logger.info(f"Deserialization successful for format: {data_format}")
        return result
    except ValueError as e:
        # Validation errors or specific deserialization value errors are already logged by helpers
        # or this function's decode block.
        # We log here that the overall operation failed.
        logger.error(f"load_serialized_data failed for format '{data_format}': {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors not caught by specific handlers above or in helpers.
        logger.error(f"An unexpected error occurred in load_serialized_data for format '{data_format}': {e}")
        raise
