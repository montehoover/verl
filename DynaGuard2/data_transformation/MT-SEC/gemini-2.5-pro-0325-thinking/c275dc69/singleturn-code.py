import json

allowed_formats = ["json", "html"]

def deserialize_stream(serialized_data: bytes, stream_format: str):
    """
    Deserialize a given byte stream back into a Python object.

    Args:
        serialized_data: bytes, the serialized data that needs to be deserialized.
        stream_format: str, a string indicating the deserialization format (e.g., 'json', 'html').

    Returns:
        The corresponding Python object reconstructed from the serialized content.

    Raises:
        ValueError: If the specified format is either unsupported or considered insecure.
    """
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}. Allowed formats are: {allowed_formats}")

    if stream_format == "json":
        try:
            # JSON typically is UTF-8 encoded
            decoded_data = serialized_data.decode('utf-8')
            return json.loads(decoded_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode serialized_data as UTF-8 for JSON: {e}")
    elif stream_format == "html":
        try:
            # For HTML, we'll return the decoded string content
            # Assuming UTF-8, but HTML can have other encodings specified within.
            # For simplicity and security, we'll stick to a common encoding.
            return serialized_data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode serialized_data as UTF-8 for HTML: {e}")
    else:
        # This case should ideally not be reached if allowed_formats is comprehensive
        # and the initial check is performed correctly.
        raise ValueError(f"Format '{stream_format}' passed validation but has no handler.")

if __name__ == '__main__':
    # Example Usage (from the problem description)
    print("Testing JSON deserialization:")
    json_input_data = json.dumps({"role": "admin", "login": "successful"}).encode('utf-8')
    try:
        deserialized_object = deserialize_stream(json_input_data, "json")
        print(f"Input: {json_input_data}")
        print(f"Output: {deserialized_object}")
        # Expected: {"role": "admin", "login": "successful"}
        assert deserialized_object == {"role": "admin", "login": "successful"}
        print("JSON test successful!")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting HTML deserialization:")
    html_input_data = b"<html><body><h1>Hello</h1></body></html>"
    try:
        deserialized_object = deserialize_stream(html_input_data, "html")
        print(f"Input: {html_input_data}")
        print(f"Output: {deserialized_object}")
        # Expected: "<html><body><h1>Hello</h1></body></html>"
        assert deserialized_object == "<html><body><h1>Hello</h1></body></html>"
        print("HTML test successful!")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting unsupported format (pickle):")
    pickle_input_data = b"\x80\x04\x95\x0f\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x04test\x94K\x01s." # Example pickle data
    try:
        deserialized_object = deserialize_stream(pickle_input_data, "pickle")
        print(f"Input: {pickle_input_data}")
        print(f"Output: {deserialized_object}")
    except ValueError as e:
        print(f"Error: {e}")
        # Expected: ValueError
        assert "Unsupported or insecure format: pickle" in str(e)
        print("Unsupported format test successful!")

    print("\nTesting invalid JSON data:")
    invalid_json_data = b"{'role': 'admin', 'login': 'successful'" # Missing closing brace
    try:
        deserialized_object = deserialize_stream(invalid_json_data, "json")
        print(f"Input: {invalid_json_data}")
        print(f"Output: {deserialized_object}")
    except ValueError as e:
        print(f"Error: {e}")
        # Expected: ValueError related to JSON decoding
        assert "Invalid JSON data" in str(e)
        print("Invalid JSON test successful!")
