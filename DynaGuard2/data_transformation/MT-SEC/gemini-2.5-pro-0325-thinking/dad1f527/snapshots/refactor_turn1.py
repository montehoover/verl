import json

allowed_serialized_formats = ["json", "text"]

def load_serialized_data(filepath: str, data_format: str):
    """
    Processes serialized data from a file, adhering to secure deserialization practices.

    Args:
        filepath: str, path to the serialized file.
        data_format: str, defines the format of the serialized data.
                     Must be one from the supported safe formats.

    Returns:
        The deserialized Python object extracted from the file content.

    Raises:
        ValueError: When the format is untrusted or prohibited.
        FileNotFoundError: If the specified filepath does not exist.
        IOError: If there's an issue reading the file.
        json.JSONDecodeError: If data_format is 'json' and the file content is not valid JSON.
    """
    if data_format not in allowed_serialized_formats:
        raise ValueError(
            f"Unsupported data format: {data_format}. Allowed formats are: {allowed_serialized_formats}"
        )

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if data_format == "json":
                return json.load(f)
            elif data_format == "text":
                return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")
    except IOError as e:
        raise IOError(f"Error reading file '{filepath}': {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from file '{filepath}': {e.msg}", e.doc, e.pos)

if __name__ == '__main__':
    # Example Usage (optional - for testing purposes)

    # Create dummy files for testing
    with open("data.json", "w") as f:
        json.dump({"name": "test", "value": 123}, f)

    with open("data.txt", "w") as f:
        f.write("This is a test text file.")

    with open("data.xml", "w") as f: # Untrusted format
        f.write("<note><to>User</to><from>Dev</from><heading>Reminder</heading></note>")

    print("Testing with allowed formats:")
    try:
        json_data = load_serialized_data("data.json", "json")
        print(f"Successfully loaded JSON data: {json_data}")
    except Exception as e:
        print(f"Error loading JSON: {e}")

    try:
        text_data = load_serialized_data("data.txt", "text")
        print(f"Successfully loaded text data: '{text_data}'")
    except Exception as e:
        print(f"Error loading text: {e}")

    print("\nTesting with an untrusted format:")
    try:
        xml_data = load_serialized_data("data.xml", "xml")
        print(f"Loaded XML data: {xml_data}")
    except ValueError as e:
        print(f"Caught expected error for XML: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for XML: {e}")

    print("\nTesting with a non-existent file:")
    try:
        data = load_serialized_data("nonexistent.json", "json")
        print(f"Loaded data: {data}")
    except FileNotFoundError as e:
        print(f"Caught expected error for non-existent file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for non-existent file: {e}")

    # Create a malformed JSON file for testing json.JSONDecodeError
    with open("malformed.json", "w") as f:
        f.write('{"name": "test", "value": 123') # Missing closing brace

    print("\nTesting with malformed JSON:")
    try:
        malformed_json_data = load_serialized_data("malformed.json", "json")
        print(f"Successfully loaded malformed JSON data: {malformed_json_data}")
    except json.JSONDecodeError as e:
        print(f"Caught expected error for malformed JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for malformed JSON: {e}")
