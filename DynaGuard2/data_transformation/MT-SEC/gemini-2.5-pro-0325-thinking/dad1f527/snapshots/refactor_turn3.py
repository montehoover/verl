import json
import logging
import os

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to write logs to a file
log_file_path = os.path.join(os.getcwd(), "deserialization.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers: # Avoid adding multiple handlers if script is re-run in some environments
    logger.addHandler(file_handler)

allowed_serialized_formats = ["json", "text"]

def _validate_data_format(data_format: str, allowed_formats: list):
    """
    Checks if the provided data format is allowed.

    Args:
        data_format: The format string to validate.
        allowed_formats: A list of allowed format strings.

    Raises:
        ValueError: If the data_format is not in allowed_formats.
    """
    if data_format not in allowed_formats:
        raise ValueError(
            f"Unsupported data format: {data_format}. Allowed formats are: {allowed_formats}"
        )

def _deserialize_content(content: str, data_format: str):
    """
    Deserializes the string content based on the given format.

    Args:
        content: The string content to deserialize.
        data_format: The data format (e.g., "json", "text").
                     It's assumed this format has already been validated.

    Returns:
        The deserialized data.

    Raises:
        json.JSONDecodeError: If data_format is 'json' and content is not valid JSON.
    """
    if data_format == "json":
        return json.loads(content)
    elif data_format == "text":
        return content
    # No 'else' branch is needed here because data_format is expected to be
    # validated by _validate_data_format before this function is called.


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
    logger.info(f"Attempting to load and deserialize file: '{filepath}' with format: '{data_format}'")
    try:
        # Validate the data format first
        _validate_data_format(data_format, allowed_serialized_formats)

        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # Deserialize the content using the appropriate helper
        # The _deserialize_content function may raise json.JSONDecodeError
        deserialized_data = _deserialize_content(file_content, data_format)
        logger.info(f"Successfully deserialized file: '{filepath}' with format: '{data_format}'")
        return deserialized_data

    except ValueError as e:
        logger.error(f"Format validation error for '{filepath}' (format: {data_format}): {e}")
        raise
    except FileNotFoundError:
        # Specific exception for file not found
        logger.error(f"File not found: '{filepath}'")
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")
    except IOError as e:
        # General IO exception for other file reading issues
        logger.error(f"IOError reading file '{filepath}': {e}")
        raise IOError(f"Error reading file '{filepath}': {e}")
    except json.JSONDecodeError as e:
        # Re-raise json.JSONDecodeError with filepath context.
        # This error originates from json.loads() within _deserialize_content.
        logger.error(f"JSON decoding error for file '{filepath}': {e.msg} (document: '{e.doc}', position: {e.pos})")
        raise json.JSONDecodeError(f"Error decoding JSON from file '{filepath}': {e.msg}", e.doc, e.pos)
    except Exception as e:
        logger.critical(f"An unexpected error occurred while processing file '{filepath}' with format '{data_format}': {e}", exc_info=True)
        raise

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
