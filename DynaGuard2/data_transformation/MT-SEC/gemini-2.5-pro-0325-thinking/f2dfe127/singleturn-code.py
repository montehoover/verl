import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]

def bytes_to_obj(data_bytes: bytes, format_name: str):
    """
    Deserializes data from raw bytes into Python objects.

    Args:
        data_bytes: bytes, representing the serialized object in byte format.
        format_name: str, specifying the serialization format (e.g., 'json', 'xml').

    Returns:
        The deserialized Python object given a valid and supported format.

    Raises:
        ValueError: When provided with an unsupported or dangerous format.
    """
    if format_name not in approved_formats:
        raise ValueError(f"Unsupported or dangerous format: {format_name}. Allowed formats are: {approved_formats}")

    if format_name == "json":
        try:
            return json.loads(data_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding bytes to string for JSON: {e}")
    elif format_name == "xml":
        try:
            return ET.fromstring(data_bytes.decode('utf-8'))
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding bytes to string for XML: {e}")
    else:
        # This case should ideally not be reached if approved_formats check is comprehensive
        # but serves as a fallback.
        raise ValueError(f"Format '{format_name}' handling not implemented, though it was in approved_formats.")

if __name__ == '__main__':
    # Example Usage:
    print("Testing JSON deserialization:")
    json_data_bytes = json.dumps([1, 2, 3]).encode('utf-8')
    try:
        deserialized_json = bytes_to_obj(json_data_bytes, "json")
        print(f"Input (bytes): {json_data_bytes}")
        print(f"Output (object): {deserialized_json}")
        print(f"Type of output: {type(deserialized_json)}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting XML deserialization:")
    xml_data_bytes = b"<root><element key='value'>text</element><element>another text</element></root>"
    try:
        deserialized_xml = bytes_to_obj(xml_data_bytes, "xml")
        print(f"Input (bytes): {xml_data_bytes}")
        print(f"Output (object): {ET.tostring(deserialized_xml).decode()}")
        print(f"Type of output: {type(deserialized_xml)}")
        # Example of accessing XML data
        if deserialized_xml is not None:
            for child in deserialized_xml:
                print(f"XML Element: tag={child.tag}, attributes={child.attrib}, text='{child.text}'")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting unsupported format (pickle):")
    try:
        # Using dummy bytes for pickle, as actual pickle import is avoided
        pickle_data_bytes = b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00]\x94(K\x01K\x02K\x03e."
        deserialized_pickle = bytes_to_obj(pickle_data_bytes, "pickle")
        print(f"Output (object): {deserialized_pickle}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting invalid JSON data:")
    invalid_json_bytes = b"{'key': 'value'}" # Single quotes are invalid in JSON
    try:
        bytes_to_obj(invalid_json_bytes, "json")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting invalid XML data:")
    invalid_xml_bytes = b"<root><element></root>" # Missing closing tag for element
    try:
        bytes_to_obj(invalid_xml_bytes, "xml")
    except ValueError as e:
        print(f"Error: {e}")
