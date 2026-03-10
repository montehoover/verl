import json
import csv
import io
import logging

# Configure logger
logger = logging.getLogger(__name__)

approved_formats = ["json", "csv"]

def _validate_format(format_type: str):
    """
    Validates if the serialization format is approved.

    Args:
        format_type: str, the format used for serialization.

    Raises:
        ValueError: if an unsupported or insecure serialization method is used.
    """
    if format_type not in approved_formats:
        raise ValueError(
            f"Unsupported or insecure serialization format: {format_type}. "
            f"Approved formats are: {', '.join(approved_formats)}"
        )

def _deserialize_json_data(raw_data: bytes):
    """
    Deserializes JSON data.

    Args:
        raw_data: bytes, representing the JSON serialized input stream.

    Returns:
        The deserialized Python object from JSON.

    Raises:
        ValueError: if JSON data is invalid or decoding fails.
    """
    try:
        # json.loads expects a string, so decode bytes to string
        decoded_data = raw_data.decode('utf-8')
        return json.loads(decoded_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding raw_data as UTF-8 for JSON: {e}")

def _deserialize_csv_data(raw_data: bytes):
    """
    Deserializes CSV data.

    Args:
        raw_data: bytes, representing the CSV serialized input stream.

    Returns:
        A list of lists representing the CSV data.

    Raises:
        ValueError: if CSV data is invalid or decoding fails.
    """
    try:
        # csv.reader expects a text-based file-like object
        decoded_data = raw_data.decode('utf-8')
        # Use io.StringIO to treat the string as a file
        csv_file = io.StringIO(decoded_data)
        reader = csv.reader(csv_file)
        # Convert reader object to a list of lists
        return list(reader)
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding raw_data as UTF-8 for CSV: {e}")
    except csv.Error as e:
        raise ValueError(f"Invalid CSV data: {e}")

def deserialize_object(raw_data: bytes, format_type: str):
    """
    Deserialize input data into a Python object.

    Args:
        raw_data: bytes, representing the serialized input stream.
        format_type: str, the format used for serialization (e.g. 'json', 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: if an unsupported or insecure serialization method is used.
    """
    logger.info(f"Attempting to deserialize data with format: {format_type}")
    try:
        _validate_format(format_type)

        if format_type == "json":
            result = _deserialize_json_data(raw_data)
        elif format_type == "csv":
            result = _deserialize_csv_data(raw_data)
        else:
            # This case should ideally be caught by _validate_format,
            # but as a safeguard for logic errors (e.g. new approved format without implementation branch):
            err_msg = f"Deserialization for format '{format_type}' not implemented, though it is in approved_formats."
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        logger.info(f"Successfully deserialized data with format: {format_type}")
        return result
    except ValueError as e:
        logger.error(f"Deserialization failed for format '{format_type}': {e}")
        raise
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during deserialization for format '{format_type}': {e}", exc_info=True)
        raise

if __name__ == '__main__':
    # Basic logging configuration for console output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example Usage:

    # JSON example
    json_data_bytes = b'{"name": "John Doe", "age": 30, "city": "New York"}'
    try:
        deserialized_json = deserialize_object(json_data_bytes, "json")
        print("Deserialized JSON:", deserialized_json)
    except ValueError as e:
        print(f"Error: {e}")

    print("-" * 20)

    # CSV example
    csv_data_bytes = b'name,age,city\nAlice,25,London\nBob,32,Paris'
    try:
        deserialized_csv = deserialize_object(csv_data_bytes, "csv")
        print("Deserialized CSV:", deserialized_csv)
    except ValueError as e:
        print(f"Error: {e}")

    print("-" * 20)

    # Pickle example (should fail)
    pickle_data_bytes = b'\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x04name\x94\x8c\x08John Doe\x94s.' # A dummy pickle string
    try:
        deserialized_pickle = deserialize_object(pickle_data_bytes, "pickle")
        print("Deserialized Pickle:", deserialized_pickle)
    except ValueError as e:
        print(f"Error (Pickle): {e}")

    print("-" * 20)

    # Invalid JSON example
    invalid_json_data_bytes = b'{"name": "Jane Doe", "age": "thirty"}' # age should be int
    try:
        deserialized_invalid_json = deserialize_object(invalid_json_data_bytes, "json")
        print("Deserialized Invalid JSON:", deserialized_invalid_json) # This will print if JSON is valid
    except ValueError as e:
        print(f"Error (Invalid JSON): {e}") # This should not be hit by this specific example, as "thirty" is a valid JSON string.
                                            # A better example for invalid JSON would be malformed syntax.

    invalid_json_syntax_bytes = b'{"name": "Jane Doe", "age": 30,' # Missing closing brace
    try:
        deserialized_invalid_json_syntax = deserialize_object(invalid_json_syntax_bytes, "json")
        print("Deserialized Invalid JSON Syntax:", deserialized_invalid_json_syntax)
    except ValueError as e:
        print(f"Error (Invalid JSON Syntax): {e}")

    print("-" * 20)
    
    # Invalid UTF-8 for JSON
    invalid_utf8_bytes = b'\xff\xfe{"name": "test"}' # BOM for UTF-16, invalid for UTF-8 start
    try:
        deserialized_invalid_utf8_json = deserialize_object(invalid_utf8_bytes, "json")
        print("Deserialized Invalid UTF-8 JSON:", deserialized_invalid_utf8_json)
    except ValueError as e:
        print(f"Error (Invalid UTF-8 JSON): {e}")

    print("-" * 20)

    # Invalid UTF-8 for CSV
    try:
        deserialized_invalid_utf8_csv = deserialize_object(invalid_utf8_bytes, "csv")
        print("Deserialized Invalid UTF-8 CSV:", deserialized_invalid_utf8_csv)
    except ValueError as e:
        print(f"Error (Invalid UTF-8 CSV): {e}")
