import json

allowed_formats = ["json", "text"]

def process_serialfile(input_path: str, format_type: str):
    """
    Processes serialized data from a file, adhering to secure deserialization practices.

    Args:
        input_path: str, path to the serialized file.
        format_type: str, defines the format of the serialized data.
                     Must be one from the supported safe formats.

    Returns:
        The deserialized Python object extracted from the file content.

    Raises:
        ValueError: When the format is untrusted or prohibited.
        FileNotFoundError: If the input_path does not exist.
        IOError: If there's an issue reading the file.
        json.JSONDecodeError: If format_type is 'json' and the file is not valid JSON.
    """
    if format_type not in allowed_formats:
        raise ValueError(
            f"Unsupported format_type: '{format_type}'. "
            f"Allowed formats are: {', '.join(allowed_formats)}"
        )

    with open(input_path, 'r') as f:
        if format_type == "json":
            return json.load(f)
        elif format_type == "text":
            return f.read()
        # This part should ideally not be reached due to the check above,
        # but as a safeguard or for future extension:
        else:
            # This case should be prevented by the initial format check.
            # Including for robustness, though theoretically unreachable
            # if allowed_formats is strictly enforced.
            raise ValueError(f"Internal error: Format '{format_type}' was allowed but not handled.")

if __name__ == '__main__':
    # Example Usage (optional - for testing purposes)
    # Create dummy files for testing
    with open("data.json", "w") as f:
        json.dump({"name": "test", "value": 123}, f)

    with open("data.txt", "w") as f:
        f.write("This is a test text file.")

    with open("data.pickle", "w") as f: # Unsafe format example
        f.write("This is a pickle file (not really, just for filename).")

    print("Testing with JSON:")
    try:
        data_json = process_serialfile("data.json", "json")
        print(f"Successfully processed JSON: {data_json}")
    except Exception as e:
        print(f"Error processing JSON: {e}")

    print("\nTesting with TEXT:")
    try:
        data_text = process_serialfile("data.txt", "text")
        print(f"Successfully processed TEXT: '{data_text}'")
    except Exception as e:
        print(f"Error processing TEXT: {e}")

    print("\nTesting with unsupported format (pickle):")
    try:
        data_pickle = process_serialfile("data.pickle", "pickle")
        print(f"Successfully processed PICKLE: {data_pickle}") # Should not happen
    except ValueError as e:
        print(f"Correctly caught ValueError for pickle: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for pickle: {e}")

    print("\nTesting with non-existent file:")
    try:
        process_serialfile("nonexistent.json", "json")
    except FileNotFoundError as e:
        print(f"Correctly caught FileNotFoundError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for non-existent file: {e}")
