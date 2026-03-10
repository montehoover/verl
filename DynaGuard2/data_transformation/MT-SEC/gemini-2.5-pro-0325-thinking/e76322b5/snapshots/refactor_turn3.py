import json
import csv
import io
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

supported_formats = ["json", "csv"]


def _deserialize_json(encoded_bytes: bytes):
    """Deserialize JSON encoded bytes."""
    try:
        decoded_string = encoded_bytes.decode('utf-8')
        data = json.loads(decoded_string)
        logger.info("Successfully deserialized JSON data.")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to deserialize JSON data: Invalid JSON data: {e}")
        raise ValueError(f"Invalid JSON data: {e}")
    except UnicodeDecodeError as e:
        logger.error(f"Failed to deserialize JSON data: Error decoding bytes to string: {e}")
        raise ValueError(f"Error decoding bytes to string for JSON: {e}")


def _deserialize_csv(encoded_bytes: bytes):
    """Deserialize CSV encoded bytes."""
    try:
        decoded_string = encoded_bytes.decode('utf-8')
        # Use io.StringIO to treat the string as a file
        csv_file = io.StringIO(decoded_string)
        reader = csv.reader(csv_file)
        # Convert reader to a list of lists
        data = [row for row in reader]
        logger.info("Successfully deserialized CSV data.")
        return data
    except UnicodeDecodeError as e:
        logger.error(f"Failed to deserialize CSV data: Error decoding bytes to string: {e}")
        raise ValueError(f"Error decoding bytes to string for CSV: {e}")
    except csv.Error as e:
        logger.error(f"Failed to deserialize CSV data: Invalid CSV data: {e}")
        raise ValueError(f"Invalid CSV data: {e}")


def decode_data(encoded_bytes: bytes, format_name: str):
    """
    Deserialize input data into a Python object.

    Args:
        encoded_bytes: bytes, representing the serialized input stream.
        format_name: str, the format used for serialization (e.g. 'json', 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: if an unsupported or insecure serialization method is used.
    """
    logger.info(f"Attempting to decode data with format: {format_name}")
    if format_name not in supported_formats:
        error_message = (
            f"Unsupported or insecure serialization format: {format_name}. "
            f"Supported formats are: {', '.join(supported_formats)}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        if format_name == "json":
            result = _deserialize_json(encoded_bytes)
            logger.info(f"Successfully decoded data using format: {format_name}")
            return result
        elif format_name == "csv":
            result = _deserialize_csv(encoded_bytes)
            logger.info(f"Successfully decoded data using format: {format_name}")
            return result
    except ValueError as e:
        # Logging for errors raised from _deserialize_json or _deserialize_csv
        # The specific error is already logged in the helper functions.
        # Here we log the failure of the decode_data operation itself.
        logger.error(f"Failed to decode data using format {format_name}: {e}")
        raise # Re-raise the ValueError caught from helper functions
    
    # This part should ideally not be reached if format_name is validated correctly
    # but as a fallback:
    # Defensive coding: although format_name is checked against supported_formats,
    # an explicit else or further check ensures clarity if more formats are added
    # or if logic changes.
    raise ValueError(f"Deserialization for format '{format_name}' not implemented after validation.")
