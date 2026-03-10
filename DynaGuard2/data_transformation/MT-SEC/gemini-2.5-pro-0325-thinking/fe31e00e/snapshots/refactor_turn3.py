import json
import configparser
import logging
import os

# --- Logger Setup ---
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file_path = os.path.join(os.getcwd(), "deserialization.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
# --- End Logger Setup ---

# a controlled list of formats that are considered safe for deserialization operations
trusted_formats = ["json", "ini"]

def _validate_serialization_format(serialization_format: str, allowed_formats: list):
    """
    Validates if the serialization format is in the list of allowed formats.

    Args:
        serialization_format: The format to validate.
        allowed_formats: A list of allowed format strings.

    Raises:
        ValueError: If the serialization_format is not in allowed_formats.
    """
    if serialization_format not in allowed_formats:
        error_message = (
            f"Unsupported serialization format: {serialization_format}. "
            f"Allowed formats are: {', '.join(allowed_formats)}"
        )
        logger.error(f"Validation failed: {error_message}")
        raise ValueError(error_message)

def _deserialize_json_data(string_data: str):
    """
    Deserializes JSON string data.

    Args:
        string_data: The JSON string to deserialize.

    Returns:
        The Python object after successful JSON deserialization.
    """
    return json.loads(string_data)

def _deserialize_ini_data(string_data: str):
    """
    Deserializes INI string data into a dictionary.

    Args:
        string_data: The INI string to deserialize.

    Returns:
        A dictionary representation of the INI data.
    """
    config = configparser.ConfigParser()
    config.read_string(string_data)
    output_dict = {section: dict(config.items(section)) for section in config.sections()}
    # Include default section items if they exist and are not already in a specific section
    if config.defaults():
        default_items = {k: v for k, v in config.defaults().items() if k not in output_dict}
        if default_items : # Check if default_items is not empty
            if 'DEFAULT' not in output_dict and any(config.defaults()):
                 output_dict['DEFAULT'] = dict(config.defaults())
            elif 'DEFAULT' in output_dict and any(config.defaults()): # Merge if DEFAULT section exists
                pass # Explicit section already captured
            elif any(config.defaults()): # No 'DEFAULT' section, but defaults exist
                output_dict['DEFAULT'] = dict(config.defaults())
    return output_dict

def securely_load_data(byte_data: bytes, serialization_format: str):
    """
    Deserialize incoming binary data safely using only permitted formats.

    Args:
        byte_data: bytes, serialized data received in binary format.
        serialization_format: str, a string specifying the mechanism used to
                              serialize the data (e.g., 'json', 'ini').

    Returns:
        The Python object after successful deserialization.

    Raises:
        ValueError: If the serialization_format is not in trusted_formats.
        NotImplementedError: If a trusted format handler is not implemented.
        # Other exceptions from underlying libraries (e.g., json.JSONDecodeError)
    """
    logger.info(f"Attempting to deserialize data with format: {serialization_format}")
    try:
        _validate_serialization_format(serialization_format, trusted_formats)

        # Assuming UTF-8 encoding for string-based formats like JSON and INI
        # This might need to be a parameter or detected if other encodings are possible.
        try:
            string_data = byte_data.decode('utf-8')
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode byte_data as UTF-8 for format {serialization_format}: {e}")
            raise ValueError(f"Failed to decode byte_data as UTF-8: {e}")

        deserialized_object = None
        if serialization_format == "json":
            deserialized_object = _deserialize_json_data(string_data)
        elif serialization_format == "ini":
            deserialized_object = _deserialize_ini_data(string_data)
        else:
            # This case should ideally not be reached if trusted_formats check is exhaustive
            # and all trusted formats are implemented.
            # However, it's good practice to have a fallback.
            # Validation should catch this, but as a safeguard:
            err_msg = f"Deserialization for trusted format '{serialization_format}' is not implemented."
            logger.error(err_msg)
            raise NotImplementedError(err_msg)
        
        logger.info(f"Successfully deserialized data with format: {serialization_format}")
        return deserialized_object

    except ValueError as ve: # Catch validation errors or decoding errors
        logger.error(f"Deserialization failed for format {serialization_format}: {ve}")
        raise
    except NotImplementedError as nie: # Catch not implemented for trusted format
        logger.error(f"Deserialization failed for format {serialization_format}: {nie}")
        raise
    except Exception as e: # Catch other deserialization errors (e.g., json.JSONDecodeError)
        logger.error(f"Deserialization failed for format {serialization_format} with an unexpected error: {e}")
        # Re-raise the original exception to maintain existing behavior for specific errors
        # like json.JSONDecodeError, configparser.Error, etc.
        raise

if __name__ == '__main__':
    # Example Usage:

    # JSON Example
    json_byte_data = b'{"name": "test", "value": 123}'
    try:
        deserialized_json = securely_load_data(json_byte_data, "json")
        print("Deserialized JSON:", deserialized_json)
    except Exception as e:
        print(f"JSON Error: {e}")

    json_byte_data_invalid = b'{"name": "test", "value": 123,}' # trailing comma
    try:
        deserialized_json = securely_load_data(json_byte_data_invalid, "json")
        print("Deserialized JSON (invalid):", deserialized_json)
    except Exception as e:
        print(f"JSON Error (invalid input): {e}")

    # INI Example
    ini_byte_data = b"""
[Section1]
key1 = value1
key2 = 100

[Section2]
option = true
"""
    try:
        deserialized_ini = securely_load_data(ini_byte_data, "ini")
        print("Deserialized INI:", deserialized_ini)
    except Exception as e:
        print(f"INI Error: {e}")

    # Pickle Example (should fail)
    pickle_byte_data = b'\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x04name\x94\x8c\x04test\x94s.'
    try:
        deserialized_pickle = securely_load_data(pickle_byte_data, "pickle")
        print("Deserialized Pickle:", deserialized_pickle)
    except Exception as e:
        print(f"Pickle Error: {e}")

    # Unsupported format example
    xml_byte_data = b"<data><item>test</item></data>"
    try:
        deserialized_xml = securely_load_data(xml_byte_data, "xml")
        print("Deserialized XML:", deserialized_xml)
    except Exception as e:
        print(f"XML Error: {e}")
    
    # Invalid byte data (not UTF-8)
    invalid_bytes = b'\xff\xfe\x00\x00h\x00e\x00l\x00l\x00o\x00' # UTF-16LE BOM with "hello"
    try:
        deserialized_data = securely_load_data(invalid_bytes, "json")
        print("Deserialized Data (invalid bytes):", deserialized_data)
    except Exception as e:
        print(f"Invalid Bytes Error: {e}")
