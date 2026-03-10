import json
import configparser
from io import StringIO

secure_serialization_formats = ["json", "ini"]

def process_serialized_content(payload_bytes: bytes, schema_type: str):
    """
    Deserializes binary data from designated safe formats.

    Args:
        payload_bytes: bytes, representing the serialized data in binary form.
        schema_type: str, describing the format to use for deserializing 
                     such as 'json', 'ini'.

    Returns:
        The Python object after performing the deserialization.

    Raises:
        ValueError: If the format is either unsupported or unsafe.
        TypeError: If payload_bytes is not bytes.
    """
    if not isinstance(payload_bytes, bytes):
        raise TypeError("payload_bytes must be of type bytes")

    if schema_type not in secure_serialization_formats:
        raise ValueError(f"Unsupported or unsafe schema type: {schema_type}. "
                         f"Allowed formats are: {', '.join(secure_serialization_formats)}")

    try:
        payload_str = payload_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode payload_bytes as UTF-8: {e}")


    if schema_type == "json":
        try:
            return json.loads(payload_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON content: {e}")
    elif schema_type == "ini":
        try:
            config = configparser.ConfigParser()
            # StringIO is used because read_string expects a string, not bytes
            config.read_string(payload_str)
            # Convert ConfigParser object to a dictionary for a more standard return type
            return {section: dict(config.items(section)) for section in config.sections()}
        except configparser.Error as e:
            raise ValueError(f"Failed to deserialize INI content: {e}")
    
    # This part should ideally not be reached if schema_type is validated against secure_serialization_formats
    # but as a safeguard:
    else:
        # This case should be caught by the initial check, but included for robustness
        raise ValueError(f"Internal error: Schema type {schema_type} passed validation but has no handler.")

if __name__ == '__main__':
    # Example Usage:
    
    # JSON example
    json_payload = json.dumps({"username": "admin", "status": "active"}).encode('utf-8')
    try:
        json_data = process_serialized_content(json_payload, "json")
        print("Deserialized JSON:", json_data)
    except (ValueError, TypeError) as e:
        print(f"Error processing JSON: {e}")

    # INI example
    ini_payload = "[user]\nusername = testuser\nemail = test@example.com\n\n[settings]\ntheme = dark".encode('utf-8')
    try:
        ini_data = process_serialized_content(ini_payload, "ini")
        print("Deserialized INI:", ini_data)
    except (ValueError, TypeError) as e:
        print(f"Error processing INI: {e}")

    # Pickle example (unsafe, should raise error)
    try:
        # Attempting to use an unsafe format
        pickle_payload = b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x04test\x94\x8c\x05hello\x94s."
        data = process_serialized_content(pickle_payload, "pickle")
        print("Deserialized Pickle (should not happen):", data)
    except (ValueError, TypeError) as e:
        print(f"Error processing Pickle (expected): {e}")

    # Invalid payload for JSON
    invalid_json_payload = b"{'key': 'value'}" # single quotes are invalid in JSON
    try:
        json_data = process_serialized_content(invalid_json_payload, "json")
        print("Deserialized Invalid JSON (should not happen):", json_data)
    except (ValueError, TypeError) as e:
        print(f"Error processing invalid JSON (expected): {e}")

    # Non-bytes payload
    try:
        data = process_serialized_content("not bytes", "json")
        print("Deserialized non-bytes (should not happen):", data)
    except (ValueError, TypeError) as e:
        print(f"Error processing non-bytes payload (expected): {e}")
        
    # Empty INI payload
    empty_ini_payload = b""
    try:
        ini_data = process_serialized_content(empty_ini_payload, "ini")
        print("Deserialized Empty INI:", ini_data) # Should be {}
    except (ValueError, TypeError) as e:
        print(f"Error processing empty INI: {e}")
