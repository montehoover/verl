import json
import configparser

# a controlled list of formats that are considered safe for deserialization operations
trusted_formats = ["json", "ini"]

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
    if serialization_format not in trusted_formats:
        raise ValueError(
            f"Unsupported serialization format: {serialization_format}. "
            f"Allowed formats are: {', '.join(trusted_formats)}"
        )

    # Assuming UTF-8 encoding for string-based formats like JSON and INI
    # This might need to be a parameter or detected if other encodings are possible.
    try:
        string_data = byte_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode byte_data as UTF-8: {e}")


    if serialization_format == "json":
        return json.loads(string_data)
    elif serialization_format == "ini":
        config = configparser.ConfigParser()
        config.read_string(string_data)
        # Convert ConfigParser object to a more standard dict representation
        # This makes the return type more consistent across formats.
        # If the raw ConfigParser object is preferred, return `config` directly.
        output_dict = {section: dict(config.items(section)) for section in config.sections()}
        # Include default section items if they exist and are not already in a specific section
        if config.defaults():
            default_items = {k: v for k, v in config.defaults().items() if k not in output_dict}
            if default_items : # Check if default_items is not empty
                 # if 'DEFAULT' not in output_dict and config.defaults(): # original logic
                 # output_dict['DEFAULT'] = dict(config.defaults()) # original logic
                 # Corrected logic to handle DEFAULT section items that might not be under a 'DEFAULT' key
                 # if the config object itself has defaults not tied to a specific section header.
                 # However, standard INI files usually have sections. If there's a global default section,
                 # it's often named [DEFAULT]. config.defaults() returns these.
                 # Let's assume we want to represent it as a dictionary under a 'DEFAULT' key if it exists.
                 # Or, merge them if no section is named 'DEFAULT'.
                 # For simplicity and common representation, let's put defaults under a 'DEFAULT' key
                 # if they exist and there isn't already a section explicitly named 'DEFAULT'.
                if 'DEFAULT' not in output_dict and any(config.defaults()):
                     output_dict['DEFAULT'] = dict(config.defaults())
                elif 'DEFAULT' in output_dict and any(config.defaults()): # Merge if DEFAULT section exists
                    # This case might be complex depending on desired merge strategy.
                    # For now, let's assume explicit [DEFAULT] section overrides defaults().
                    pass # Explicit section already captured
                elif any(config.defaults()): # No 'DEFAULT' section, but defaults exist
                    # This case is less common for standard INI.
                    # Let's add them under a 'DEFAULT' key for consistency.
                    output_dict['DEFAULT'] = dict(config.defaults())


        return output_dict
    else:
        # This case should ideally not be reached if trusted_formats check is exhaustive
        # and all trusted formats are implemented.
        raise NotImplementedError(
            f"Deserialization for format '{serialization_format}' is not implemented."
        )

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
