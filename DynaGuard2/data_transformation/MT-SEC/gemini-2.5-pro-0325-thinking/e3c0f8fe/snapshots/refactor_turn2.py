import json
import csv
import io

# a registry of allowed serialization methods
trusted_serializations = ["json", "csv"]

def _deserialize_json(serialized_bytes: bytes):
    """Deserializes JSON data from bytes."""
    try:
        # Assuming UTF-8 encoding for JSON data
        decoded_string = serialized_bytes.decode('utf-8')
        return json.loads(decoded_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding bytes to string for JSON: {e}")

def _deserialize_csv(serialized_bytes: bytes):
    """Deserializes CSV data from bytes."""
    try:
        # Assuming UTF-8 encoding for CSV data
        decoded_string = serialized_bytes.decode('utf-8')
        # Use io.StringIO to treat the string as a file for csv.reader
        csv_file = io.StringIO(decoded_string)
        reader = csv.reader(csv_file)
        return list(reader)
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding bytes to string for CSV: {e}")
    except csv.Error as e:
        raise ValueError(f"Invalid CSV data: {e}")

def restore_object(serialized_bytes: bytes, serialization_method: str):
    """
    Restores a Python object from a serialized data stream.

    Args:
        serialized_bytes: bytes, the data stream to deserialize.
        serialization_method: str, identifies the format used for serialization
                              (e.g. 'json', 'csv', 'pickle').

    Returns:
        A Python object restored from the serialized data.

    Raises:
        ValueError: If the method is unrecognized or insecure.
    """
    if serialization_method not in trusted_serializations:
        raise ValueError(
            f"Unsupported or insecure serialization method: {serialization_method}. "
            f"Allowed methods are: {', '.join(trusted_serializations)}"
        )

    if serialization_method == "json":
        return _deserialize_json(serialized_bytes)
    elif serialization_method == "csv":
        return _deserialize_csv(serialized_bytes)
    # This part should ideally not be reached if the initial check is comprehensive
    # and trusted_serializations only contains implemented methods.
    # However, as a safeguard for future extensions where a method might be in
    # trusted_serializations but not yet have a corresponding block here:
    else:
        # This case implies a trusted method doesn't have a deserialization path.
        # This should ideally be caught by tests or static analysis if trusted_serializations
        # and the implemented methods diverge.
        raise ValueError(f"Internal error: No deserialization logic for trusted method {serialization_method}")

if __name__ == '__main__':
    # Example Usage:

    # JSON example
    json_data = b'{"name": "Alice", "age": 30, "city": "New York"}'
    try:
        obj_json = restore_object(json_data, "json")
        print(f"Restored from JSON: {obj_json}")
    except ValueError as e:
        print(f"Error: {e}")

    json_data_invalid = b'{"name": "Alice", "age": 30,' # Invalid JSON
    try:
        obj_json_invalid = restore_object(json_data_invalid, "json")
        print(f"Restored from JSON: {obj_json_invalid}")
    except ValueError as e:
        print(f"Error processing invalid JSON: {e}")

    # CSV example
    csv_data = b"name,age,city\nBob,25,London\nCharlie,35,Paris"
    try:
        obj_csv = restore_object(csv_data, "csv")
        print(f"Restored from CSV: {obj_csv}")
    except ValueError as e:
        print(f"Error: {e}")

    csv_data_invalid_encoding = b'\xff\xfe' # Invalid UTF-8 start
    try:
        obj_csv_invalid = restore_object(csv_data_invalid_encoding, "csv")
        print(f"Restored from CSV: {obj_csv_invalid}")
    except ValueError as e:
        print(f"Error processing CSV with invalid encoding: {e}")

    # Pickle example (should be rejected)
    pickle_data = b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x04name\x94\x8c\x05David\x94s."
    try:
        obj_pickle = restore_object(pickle_data, "pickle")
        print(f"Restored from pickle: {obj_pickle}")
    except ValueError as e:
        print(f"Error: {e}")

    # Unknown method example
    xml_data = b"<person><name>Eve</name></person>"
    try:
        obj_xml = restore_object(xml_data, "xml")
        print(f"Restored from XML: {obj_xml}")
    except ValueError as e:
        print(f"Error: {e}")
