import json

acceptable_formats = ["json", "text"]

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
        ValueError: If the format is unsafe or unsupported.
        FileNotFoundError: If the file_location does not exist.
    """
    if data_format not in acceptable_formats:
        raise ValueError(
            f"Unsupported or insecure data format: {data_format}. "
            f"Allowed formats are: {acceptable_formats}"
        )

    with open(file_location, 'r') as f:
        if data_format == "json":
            return json.load(f)
        elif data_format == "text":
            return f.read()
    # Should not be reached if data_format is in acceptable_formats
    # and handled above, but as a safeguard:
    raise ValueError(f"Unhandled acceptable format: {data_format}")

if __name__ == '__main__':
    # Example Usage (requires creating dummy files)

    # Create a dummy JSON file
    test_json_data_content = {"name": "Alice", "location": "Wonderland"}
    with open("test_json_data.json", "w") as f:
        json.dump(test_json_data_content, f)

    # Create a dummy text file
    test_text_data_content = "This is a simple text file."
    with open("test_text_data.txt", "w") as f:
        f.write(test_text_data_content)

    # Test JSON deserialization
    try:
        json_data = load_serialized_data("test_json_data.json", "json")
        print(f"Successfully loaded JSON data: {json_data}")
        assert json_data == test_json_data_content
    except Exception as e:
        print(f"Error loading JSON: {e}")

    # Test text deserialization
    try:
        text_data = load_serialized_data("test_text_data.txt", "text")
        print(f"Successfully loaded text data: '{text_data}'")
        assert text_data == test_text_data_content
    except Exception as e:
        print(f"Error loading text: {e}")

    # Test unsupported format
    try:
        load_serialized_data("test_text_data.txt", "pickle")
    except ValueError as e:
        print(f"Correctly caught unsupported format: {e}")
    except Exception as e:
        print(f"Unexpected error for unsupported format: {e}")

    # Test file not found
    try:
        load_serialized_data("non_existent_file.json", "json")
    except FileNotFoundError as e:
        print(f"Correctly caught file not found: {e}")
    except Exception as e:
        print(f"Unexpected error for file not found: {e}")
