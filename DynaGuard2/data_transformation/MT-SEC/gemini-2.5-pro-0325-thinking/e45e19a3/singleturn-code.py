import json
import xml.etree.ElementTree as ET

safe_formats = ["json", "xml"]

def deserialize_data(input_data: bytes, data_type: str):
    """
    Deserializes data from raw bytes into Python objects.

    Args:
        input_data: bytes, representing the serialized object in byte format.
        data_type: str, specifying the serialization format (e.g., 'json', 'xml').

    Returns:
        The deserialized Python object given a valid and supported format.

    Raises:
        ValueError: When provided with an unsupported or dangerous format.
    """
    if data_type not in safe_formats:
        raise ValueError(f"Unsupported or dangerous format: {data_type}. Allowed formats are: {safe_formats}")

    # Decode bytes to string before deserialization
    try:
        decoded_data = input_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode input_data as UTF-8: {e}")


    if data_type == "json":
        try:
            return json.loads(decoded_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif data_type == "xml":
        try:
            return ET.fromstring(decoded_data)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML data: {e}")
    else:
        # This case should ideally not be reached if safe_formats check is comprehensive
        # and aligned with the conditional logic.
        # However, as a safeguard:
        raise ValueError(f"Deserialization logic not implemented for format: {data_type}")

if __name__ == '__main__':
    # Example Usage:
    
    # JSON example
    json_byte_data = json.dumps({"name": "John Doe", "age": 30, "isStudent": False, "courses": [{"title": "History", "credits": 3}, {"title": "Math", "credits": 4}]}).encode('utf-8')
    try:
        deserialized_json = deserialize_data(json_byte_data, "json")
        print("Deserialized JSON:", deserialized_json)
        assert deserialized_json == {"name": "John Doe", "age": 30, "isStudent": False, "courses": [{"title": "History", "credits": 3}, {"title": "Math", "credits": 4}]}
    except ValueError as e:
        print(f"Error deserializing JSON: {e}")

    # XML example
    xml_byte_data = b'<person><name>Jane Doe</name><age>25</age><city>New York</city></person>'
    try:
        deserialized_xml = deserialize_data(xml_byte_data, "xml")
        print("Deserialized XML:", ET.tostring(deserialized_xml, encoding='unicode')) # Print XML string for verification
        # Basic assertion for XML structure
        assert deserialized_xml.tag == 'person'
        assert deserialized_xml.find('name').text == 'Jane Doe'
    except ValueError as e:
        print(f"Error deserializing XML: {e}")

    # Pickle example (unsafe, should raise ValueError)
    try:
        # Simulating pickle data - in a real scenario, this would be actual pickled bytes
        pickle_byte_data = b'\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00]\x94(K\x01K\x02K\x03e.' 
        deserialized_pickle = deserialize_data(pickle_byte_data, "pickle")
        print("Deserialized Pickle:", deserialized_pickle)
    except ValueError as e:
        print(f"Error deserializing Pickle: {e}")
        assert "Unsupported or dangerous format: pickle" in str(e)

    # Example from the problem description
    example_input_data = json.dumps([1, 2, 3]).encode()
    try:
        output = deserialize_data(example_input_data, "json")
        print("Example output:", output)
        assert output == [1, 2, 3]
    except ValueError as e:
        print(f"Error in example: {e}")

    # Example with invalid JSON data
    invalid_json_data = b'{"name": "John Doe", "age": 30,}' # Trailing comma makes it invalid
    try:
        deserialize_data(invalid_json_data, "json")
    except ValueError as e:
        print(f"Error with invalid JSON: {e}")
        assert "Invalid JSON data" in str(e)

    # Example with invalid XML data
    invalid_xml_data = b'<person><name>Jane Doe</city></person>' # Mismatched tags
    try:
        deserialize_data(invalid_xml_data, "xml")
    except ValueError as e:
        print(f"Error with invalid XML: {e}")
        assert "Invalid XML data" in str(e)
    
    # Example with non-UTF-8 data
    non_utf8_data = b'\xff\xfe\x48\x00\x65\x00\x6c\x00\x6c\x00\x6f\x00' # "Hello" in UTF-16
    try:
        deserialize_data(non_utf8_data, "json")
    except ValueError as e:
        print(f"Error with non-UTF-8 data: {e}")
        assert "Failed to decode input_data as UTF-8" in str(e)

    print("All tests passed if no assertion errors above.")
