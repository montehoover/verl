import json
import xml.etree.ElementTree as ET

def deserialize_data(input_data: bytes, data_type: str):
    """
    Deserializes byte data into a Python object based on the specified data type.

    Only allows deserialization for formats listed in safe_formats.

    Args:
        input_data: The serialized byte data.
        data_type: The format of the data ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the data_type is unsupported, dangerous, or if
                    deserialization fails for the given input_data.
    """
    safe_formats = ["json", "xml"]

    if data_type not in safe_formats:
        raise ValueError(
            f"Unsupported or dangerous data type: {data_type}. "
            f"Allowed formats are: {', '.join(safe_formats)}"
        )

    try:
        if data_type == "json":
            # Assuming UTF-8 encoding for JSON byte data
            decoded_string = input_data.decode('utf-8')
            return json.loads(decoded_string)
        elif data_type == "xml":
            # ET.fromstring expects a string or bytes.
            # If input_data is already bytes, it can be used directly.
            # If it were a string, it would need encoding first if the XML parser required bytes.
            # For simplicity, we assume input_data is bytes as per type hint.
            return ET.fromstring(input_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}") from e
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML data: {e}") from e
    except UnicodeDecodeError as e:
        raise ValueError(f"Error decoding input_data as UTF-8 for JSON: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors during deserialization
        raise ValueError(f"Deserialization failed for {data_type}: {e}") from e
