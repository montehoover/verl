import os
import json
import xml.etree.ElementTree as ET
from typing import Any, List # Added Any and List for type hinting

# Whitelist of accepted serialized formats
allowed_serialized_formats: List[str] = ["json", "text"]

def read_file_content(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If an error occurs during file reading.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        # Re-raise FileNotFoundError to be handled by the caller
        raise
    except IOError as e:
        # Handle other potential I/O errors
        # For now, re-raising a generic IOError, but could be more specific
        raise IOError(f"Error reading file {file_path}: {e}")

def validate_file_format(content: str) -> str:
    """
    Determines the format of the file content (JSON, XML, or TEXT).

    Args:
        content: The file content as a string.

    Returns:
        A string indicating the format ("JSON", "XML", or "TEXT").
        Raises ValueError if the format is unrecognized or potentially unsafe (currently, all non-JSON/XML is TEXT).
    """
    stripped_content = content.strip()

    if not stripped_content:
        return "TEXT" # Empty or whitespace-only content is considered TEXT

    # Try to parse as JSON
    try:
        json.loads(stripped_content)
        return "JSON"
    except json.JSONDecodeError:
        pass  # Not JSON

    # Try to parse as XML
    try:
        ET.fromstring(stripped_content)
        return "XML"
    except ET.ParseError:
        pass  # Not XML

    # Default to TEXT if not identifiable as JSON or XML
    # Add checks for "unsafe" or more specific "unrecognized" criteria here if needed.
    # For now, any content that is not valid JSON or XML is considered TEXT.
    # A common check for "unsafe" might involve looking for executable patterns or binary content,
    # but that is beyond the current scope of "JSON, XML, TEXT" detection.
    return "TEXT"

def load_serialized_data(filepath: str, data_format: str) -> Any:
    """
    Securely deserializes data from a file based on the specified format.

    Args:
        filepath: The path to the file.
        data_format: The format of the serialized data (e.g., "json", "text").
                     Must be in `allowed_serialized_formats`.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the data_format is untrusted or prohibited,
                    or if deserialization fails for the given format.
        FileNotFoundError: If the file does not exist (propagated from read_file_content).
        IOError: If an error occurs during file reading (propagated from read_file_content).
    """
    normalized_format = data_format.lower()
    if normalized_format not in allowed_serialized_formats:
        raise ValueError(
            f"Untrusted or prohibited data format: '{data_format}'. "
            f"Allowed formats are: {allowed_serialized_formats}"
        )

    content = read_file_content(filepath)

    if normalized_format == "json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON content from {filepath}: {e}")
    elif normalized_format == "text":
        # For "text" format, the content is already a string, which is the desired Python object.
        return content
    else:
        # This case should ideally not be reached if allowed_serialized_formats is checked correctly,
        # but as a safeguard:
        raise ValueError(f"Unsupported data format specified: {data_format}")


if __name__ == '__main__':
    # Example usage for read_file_content (optional, for testing the function)
    # Create a dummy file for testing
    dummy_file_path = "test_file.txt"
    with open(dummy_file_path, "w") as f:
        f.write("Hello, this is a test file.\n")
        f.write("It has multiple lines.")

    try:
        file_content = read_file_content(dummy_file_path)
        print("File content:\n", file_content)
    except FileNotFoundError:
        print(f"Error: The file {dummy_file_path} was not found.")
    except IOError as e:
        print(f"An I/O error occurred: {e}")
    finally:
        # Clean up the dummy file
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

    # Test with a non-existent file
    non_existent_file = "non_existent_file.txt"
    try:
        file_content = read_file_content(non_existent_file)
        print("File content:\n", file_content) # This line should not be reached
    except FileNotFoundError:
        print(f"\nError: The file {non_existent_file} was not found (as expected).")
    except IOError as e:
        print(f"\nAn I/O error occurred: {e}")

    print("\n--- Testing validate_file_format ---")
    test_contents = {
        "json_valid": ('{"name": "test", "value": 123}', "JSON"),
        "json_array_valid": ('[1, 2, 3]', "JSON"),
        "xml_valid": ('<note><to>Tove</to><from>Jani</from><heading>Reminder</heading></note>', "XML"),
        "text_simple": ('Hello world!', "TEXT"),
        "text_empty": ('', "TEXT"),
        "text_whitespace": ('   \n\t   ', "TEXT"),
        "json_malformed_keys": ("{key: 'value'}", "TEXT"), # Python dict like, not valid JSON
        "json_malformed_trailing_comma": ('{"name": "test",}', "TEXT"), # Often allowed, but not strict JSON
        "xml_malformed_unclosed_tag": ('<note><to>Tove</to>', "TEXT"),
        "text_looks_like_xml_but_isnt": ('<notxml', "TEXT"),
        "text_looks_like_json_but_isnt": ('{notjson', "TEXT"),
    }

    for name, (content_str, expected_format) in test_contents.items():
        try:
            detected_format = validate_file_format(content_str)
            print(f"Test '{name}': Expected '{expected_format}', Got '{detected_format}'. Pass: {detected_format == expected_format}")
            if detected_format != expected_format:
                print(f"   Content: \"{content_str[:50]}{'...' if len(content_str) > 50 else ''}\"")
        except ValueError as e:
            print(f"Test '{name}': Expected '{expected_format}', Got ValueError: {e}. Pass: {expected_format == 'ValueError_expected'}") # Placeholder for future
            print(f"   Content: \"{content_str[:50]}{'...' if len(content_str) > 50 else ''}\"")

    print("\n--- Testing load_serialized_data ---")

    # Create dummy files for load_serialized_data tests
    test_json_file_path = "test_data.json"
    test_text_file_path = "test_data.txt"
    test_malformed_json_file_path = "test_malformed_data.json"

    with open(test_json_file_path, "w") as f:
        json.dump({"key": "value", "number": 42}, f)

    with open(test_text_file_path, "w") as f:
        f.write("This is plain text content.\nWith multiple lines.")

    with open(test_malformed_json_file_path, "w") as f:
        f.write('{"key": "value", "number": 42,}') # Malformed JSON with trailing comma

    # Test cases for load_serialized_data
    load_tests = [
        ("valid_json", test_json_file_path, "json", {"key": "value", "number": 42}),
        ("valid_text", test_text_file_path, "text", "This is plain text content.\nWith multiple lines."),
        ("valid_JSON_caps", test_json_file_path, "JSON", {"key": "value", "number": 42}), # Test case-insensitivity
    ]

    for name, path, fmt, expected_data in load_tests:
        try:
            data = load_serialized_data(path, fmt)
            print(f"Test '{name}': Loaded data: {data}. Pass: {data == expected_data}")
            if data != expected_data:
                 print(f"   Expected: {expected_data}")
        except Exception as e:
            print(f"Test '{name}': FAILED with exception: {e}")

    # Test prohibited format
    try:
        load_serialized_data(test_text_file_path, "xml")
        print("Test 'prohibited_xml_format': FAILED - ValueError not raised.")
    except ValueError as e:
        print(f"Test 'prohibited_xml_format': Passed - Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"Test 'prohibited_xml_format': FAILED with unexpected exception: {e}")

    # Test malformed JSON
    try:
        load_serialized_data(test_malformed_json_file_path, "json")
        print("Test 'malformed_json_file': FAILED - ValueError not raised for malformed JSON.")
    except ValueError as e:
        print(f"Test 'malformed_json_file': Passed - Correctly raised ValueError for malformed JSON: {e}")
    except Exception as e:
        print(f"Test 'malformed_json_file': FAILED with unexpected exception: {e}")

    # Test non-existent file
    try:
        load_serialized_data("non_existent.json", "json")
        print("Test 'non_existent_file_load': FAILED - FileNotFoundError not raised.")
    except FileNotFoundError:
        print("Test 'non_existent_file_load': Passed - Correctly raised FileNotFoundError.")
    except Exception as e:
        print(f"Test 'non_existent_file_load': FAILED with unexpected exception: {e}")


    # Clean up dummy files
    for f_path in [test_json_file_path, test_text_file_path, test_malformed_json_file_path]:
        if os.path.exists(f_path):
            os.remove(f_path)
