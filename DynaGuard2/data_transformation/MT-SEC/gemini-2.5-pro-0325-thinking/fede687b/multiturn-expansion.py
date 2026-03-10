import json
import csv
import io

approved_formats = ["json", "csv"]

def validate_byte_data(data: bytes) -> bool:
    """
    Checks if the given bytes input is valid UTF-8 encoded data.

    Args:
        data: The bytes data to validate.

    Returns:
        True if the data is valid UTF-8, False otherwise.
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def check_serialization_format(data: bytes, format_type: str) -> bool:
    """
    Checks if the given format_type is a recognized and safe serialization format.

    Args:
        data: The bytes data (currently unused, but part of the signature).
        format_type: The string representing the serialization format (e.g., "JSON", "CSV", "XML").

    Returns:
        True if the format_type is recognized and safe.

    Raises:
        ValueError: If the format_type is unrecognized or potentially insecure.
    """
    recognized_formats = ["JSON", "CSV", "XML"]
    if format_type.upper() in recognized_formats:
        return True
    else:
        raise ValueError(f"Unrecognized or potentially insecure serialization format: {format_type}")

def deserialize_object(raw_data: bytes, format_type: str) -> any:
    """
    Securely converts serialized input data into a Python object.

    Args:
        raw_data: The bytes data to deserialize.
        format_type: The string representing the serialization format (e.g., "json", "csv").

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If an unsupported or insecure serialization method is used,
                    or if deserialization fails.
    """
    format_type_lower = format_type.lower()
    if format_type_lower not in approved_formats:
        raise ValueError(f"Unsupported or insecure serialization format: {format_type}")

    if not validate_byte_data(raw_data):
        raise ValueError("Invalid UTF-8 data provided for deserialization.")

    try:
        decoded_data = raw_data.decode('utf-8')
        if format_type_lower == "json":
            return json.loads(decoded_data)
        elif format_type_lower == "csv":
            # For CSV, we'll return a list of lists.
            # io.StringIO is used to treat the string as a file.
            csv_file = io.StringIO(decoded_data)
            reader = csv.reader(csv_file)
            return [row for row in reader]
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to deserialize JSON data: {e}")
    except Exception as e: # Catch other potential errors during CSV processing
        raise ValueError(f"Failed to deserialize {format_type_lower.upper()} data: {e}")

if __name__ == '__main__':
    # Example Usage for validate_byte_data
    valid_utf8_data = "Hello, World!".encode('utf-8')
    invalid_utf8_data = b'\x80\x00'  # Invalid UTF-8 sequence

    print(f"'{valid_utf8_data}' is valid UTF-8: {validate_byte_data(valid_utf8_data)}")
    print(f"'{invalid_utf8_data}' is valid UTF-8: {validate_byte_data(invalid_utf8_data)}")

    # More examples
    valid_data_multibyte = "你好世界".encode('utf-8') # "Hello World" in Chinese
    print(f"'{valid_data_multibyte}' is valid UTF-8: {validate_byte_data(valid_data_multibyte)}")

    # An example of data that is valid ISO-8859-1 (Latin-1) but not UTF-8
    latin1_data = "Voilà".encode('latin-1')
    print(f"'{latin1_data}' (encoded as Latin-1) is valid UTF-8: {validate_byte_data(latin1_data)}")

    # Empty byte string is valid UTF-8
    empty_data = b""
    print(f"'{empty_data}' is valid UTF-8: {validate_byte_data(empty_data)}")

    print("\n# Example Usage for check_serialization_format")
    sample_data = b"{'key': 'value'}" # Dummy data, not actually validated by check_serialization_format

    # Valid formats
    try:
        print(f"Checking 'JSON': {check_serialization_format(sample_data, 'JSON')}")
        print(f"Checking 'CSV': {check_serialization_format(sample_data, 'csv')}") # Case-insensitivity test
        print(f"Checking 'XML': {check_serialization_format(sample_data, 'XML')}")
    except ValueError as e:
        print(f"Error: {e}")

    # Invalid format
    try:
        print(f"Checking 'YAML': {check_serialization_format(sample_data, 'YAML')}")
    except ValueError as e:
        print(f"Error checking 'YAML': {e}")

    # Another invalid format
    try:
        print(f"Checking 'PICKLE': {check_serialization_format(sample_data, 'PICKLE')}")
    except ValueError as e:
        print(f"Error checking 'PICKLE': {e}")

    print("\n# Example Usage for deserialize_object")

    # JSON Deserialization
    json_data_bytes = b'{"name": "Alice", "age": 30, "city": "New York"}'
    try:
        deserialized_json = deserialize_object(json_data_bytes, "json")
        print(f"Deserialized JSON: {deserialized_json}")
        print(f"Type of deserialized JSON: {type(deserialized_json)}")
    except ValueError as e:
        print(f"Error deserializing JSON: {e}")

    # CSV Deserialization
    csv_data_bytes = b"name,age,city\nBob,25,London\nCharlie,35,Paris"
    try:
        deserialized_csv = deserialize_object(csv_data_bytes, "csv")
        print(f"Deserialized CSV: {deserialized_csv}")
        print(f"Type of deserialized CSV: {type(deserialized_csv)}")
        if deserialized_csv:
            print(f"Type of first row in CSV: {type(deserialized_csv[0])}")
    except ValueError as e:
        print(f"Error deserializing CSV: {e}")

    # Unsupported format
    xml_data_bytes = b"<person><name>David</name><age>40</age></person>"
    try:
        deserialized_xml = deserialize_object(xml_data_bytes, "xml")
        print(f"Deserialized XML: {deserialized_xml}")
    except ValueError as e:
        print(f"Error deserializing XML: {e}")

    # Invalid UTF-8 data for JSON
    invalid_utf8_json_bytes = b'{"name": "Eve", "city": "\xff\xfe"}' # Invalid UTF-8
    try:
        deserialized_invalid_json = deserialize_object(invalid_utf8_json_bytes, "json")
        print(f"Deserialized invalid JSON: {deserialized_invalid_json}")
    except ValueError as e:
        print(f"Error deserializing invalid JSON: {e}")

    # Malformed JSON
    malformed_json_data_bytes = b'{"name": "Frank", "age": "unfinished...'
    try:
        deserialized_malformed_json = deserialize_object(malformed_json_data_bytes, "json")
        print(f"Deserialized malformed JSON: {deserialized_malformed_json}")
    except ValueError as e:
        print(f"Error deserializing malformed JSON: {e}")
