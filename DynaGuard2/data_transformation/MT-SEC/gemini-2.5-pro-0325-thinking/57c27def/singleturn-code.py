import json
import configparser
from io import StringIO

# A curated list of formats that have been evaluated to ensure safe deserialization
valid_formats = ["json", "ini"]

def convert_serialized_data(raw_bytes: bytes, format_hint: str):
    """
    Deserializes binary data from external, potentially unreliable sources.

    Args:
        raw_bytes: bytes, representing the serialized data in binary form.
        format_hint: str, describing the format to use for deserializing 
                     such as 'json', 'ini'.

    Returns:
        The Python object after performing the deserialization.

    Raises:
        ValueError: If the format is either unsupported or unsafe.
        # Other exceptions like json.JSONDecodeError or configparser.Error can also be raised
        # by the respective parsers if the data is malformed.
    """
    if format_hint not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint}. Supported formats are: {valid_formats}")

    try:
        decoded_string = raw_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode raw_bytes as UTF-8: {e}")


    if format_hint == "json":
        try:
            return json.loads(decoded_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif format_hint == "ini":
        config = configparser.ConfigParser()
        try:
            config.read_string(decoded_string)
            # Convert ConfigParser object to a dictionary for a more standard Python object representation
            # This creates a dictionary where keys are section names and values are dictionaries
            # of key-value pairs within that section.
            ini_dict = {section: dict(config.items(section)) for section in config.sections()}
            # If there's a DEFAULT section and it's not empty, include it as well.
            # configparser treats DEFAULT section specially.
            if config.defaults():
                ini_dict['DEFAULT'] = dict(config.defaults())
            return ini_dict
        except configparser.Error as e: # Catches MissingSectionHeaderError, ParsingError, etc.
            raise ValueError(f"Invalid INI data: {e}")
    
    # This part should ideally not be reached if format_hint is in valid_formats
    # and handled above, but as a safeguard:
    else:
        # This case should have been caught by the initial check,
        # but it's good practice for defensive programming.
        raise ValueError(f"Internal error: Format {format_hint} was validated but not handled.")

if __name__ == '__main__':
    # Example Usage:
    
    # JSON Example
    json_data = {"username": "admin", "status": "active"}
    raw_json_bytes = json.dumps(json_data).encode('utf-8')
    
    try:
        deserialized_json = convert_serialized_data(raw_json_bytes, "json")
        print(f"Deserialized JSON: {deserialized_json}")
        assert deserialized_json == json_data
    except ValueError as e:
        print(f"Error deserializing JSON: {e}")

    # INI Example
    ini_string = """
[user]
username = editor
status = pending

[settings]
theme = dark
notifications = off
"""
    raw_ini_bytes = ini_string.encode('utf-8')
    
    try:
        deserialized_ini = convert_serialized_data(raw_ini_bytes, "ini")
        print(f"Deserialized INI: {deserialized_ini}")
        expected_ini = {
            'user': {'username': 'editor', 'status': 'pending'},
            'settings': {'theme': 'dark', 'notifications': 'off'}
        }
        assert deserialized_ini == expected_ini
    except ValueError as e:
        print(f"Error deserializing INI: {e}")

    # Pickle Example (unsafe, should raise error)
    try:
        # Simulating some pickled data (even if it's just a string for this test)
        raw_pickle_bytes = b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x08testdata\x94."
        deserialized_pickle = convert_serialized_data(raw_pickle_bytes, "pickle")
        print(f"Deserialized Pickle: {deserialized_pickle}") # Should not reach here
    except ValueError as e:
        print(f"Error deserializing Pickle (as expected): {e}")

    # Malformed JSON example
    malformed_json_bytes = b'{"username": "test", "status": "active"' # Missing closing brace
    try:
        deserialized_malformed_json = convert_serialized_data(malformed_json_bytes, "json")
        print(f"Deserialized Malformed JSON: {deserialized_malformed_json}") # Should not reach here
    except ValueError as e:
        print(f"Error deserializing malformed JSON (as expected): {e}")

    # Malformed INI example (e.g. no section header)
    malformed_ini_bytes = b'key = value'
    try:
        deserialized_malformed_ini = convert_serialized_data(malformed_ini_bytes, "ini")
        print(f"Deserialized Malformed INI: {deserialized_malformed_ini}") # Should not reach here
    except ValueError as e:
        print(f"Error deserializing malformed INI (as expected): {e}")
        
    # Empty JSON object
    empty_json_bytes = b'{}'
    try:
        deserialized_empty_json = convert_serialized_data(empty_json_bytes, "json")
        print(f"Deserialized Empty JSON: {deserialized_empty_json}")
        assert deserialized_empty_json == {}
    except ValueError as e:
        print(f"Error deserializing empty JSON: {e}")

    # Empty INI (no sections)
    empty_ini_bytes = b'' # or b'\n'
    try:
        deserialized_empty_ini = convert_serialized_data(empty_ini_bytes, "ini")
        print(f"Deserialized Empty INI: {deserialized_empty_ini}")
        assert deserialized_empty_ini == {}
    except ValueError as e:
        print(f"Error deserializing empty INI: {e}")
        
    # INI with only DEFAULT section
    default_section_ini_bytes = b'[DEFAULT]\ndefault_key = default_value'
    try:
        deserialized_default_ini = convert_serialized_data(default_section_ini_bytes, "ini")
        print(f"Deserialized INI with DEFAULT: {deserialized_default_ini}")
        expected_default_ini = {'DEFAULT': {'default_key': 'default_value'}}
        assert deserialized_default_ini == expected_default_ini
    except ValueError as e:
        print(f"Error deserializing INI with DEFAULT: {e}")
