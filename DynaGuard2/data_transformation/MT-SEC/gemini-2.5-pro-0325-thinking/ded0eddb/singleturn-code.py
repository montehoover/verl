import json
import xml.etree.ElementTree as ET

secure_formats = ["json", "xml"]

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
    if data_format not in secure_formats:
        raise ValueError(
            f"Unsupported or unsafe data format: {data_format}. "
            f"Allowed formats are: {', '.join(secure_formats)}"
        )

    # Decode bytes to string before parsing, assuming UTF-8 encoding.
    # Specific encoding might need to be handled or passed as an argument
    # if it varies.
    try:
        string_data = bytes_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode bytes_data: {e}")


    if data_format == "json":
        try:
            return json.loads(string_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif data_format == "xml":
        try:
            return ET.fromstring(string_data)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML data: {e}")
    else:
        # This case should ideally not be reached if the initial check
        # for data_format in secure_formats is comprehensive and secure_formats
        # only contains formats handled below.
        # However, as a safeguard:
        raise ValueError(f"Internal error: Unhandled secure format '{data_format}'.")

if __name__ == '__main__':
    # Example Usage:
    
    # JSON example
    json_data_bytes = json.dumps([1, 2, 3, {"key": "value"}]).encode('utf-8')
    try:
        deserialized_json = load_serialized_data(json_data_bytes, "json")
        print(f"Deserialized JSON: {deserialized_json}")
    except ValueError as e:
        print(f"Error deserializing JSON: {e}")

    # XML example
    xml_data_bytes = b"<root><element attribute='value'>text</element></root>"
    try:
        deserialized_xml = load_serialized_data(xml_data_bytes, "xml")
        if deserialized_xml is not None:
            print(f"Deserialized XML: <{deserialized_xml.tag}>")
            for child in deserialized_xml:
                print(f"  Child: <{child.tag} attribute='{child.get('attribute')}'>{child.text}</{child.tag}>")
    except ValueError as e:
        print(f"Error deserializing XML: {e}")

    # Pickle example (unsafe, should raise ValueError)
    try:
        # Note: We don't actually import pickle or create pickle data
        # as it's unsafe and not part of the allowed formats.
        # This is just to test the format validation.
        deserialized_pickle = load_serialized_data(b"some bytes", "pickle")
        print(f"Deserialized Pickle: {deserialized_pickle}")
    except ValueError as e:
        print(f"Error deserializing (pickle format test): {e}")

    # Example with invalid JSON data
    invalid_json_bytes = b"{'key': 'value',,}" # Invalid JSON syntax
    try:
        deserialized_invalid_json = load_serialized_data(invalid_json_bytes, "json")
        print(f"Deserialized invalid JSON: {deserialized_invalid_json}")
    except ValueError as e:
        print(f"Error deserializing invalid JSON: {e}")
    
    # Example with non-UTF-8 decodable bytes
    non_utf8_bytes = b'\xff\xfe\x00\x00a\x00\x00\x00' # Example non-UTF-8 bytes (UTF-16 BOM)
    try:
        deserialized_non_utf8 = load_serialized_data(non_utf8_bytes, "json")
        print(f"Deserialized non-UTF-8 data: {deserialized_non_utf8}")
    except ValueError as e:
        print(f"Error with non-UTF-8 data: {e}")
