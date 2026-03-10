import json
import xml.etree.ElementTree as ET
from typing import Any

# Allowed formats for safe deserialization
secure_formats = ["json", "xml"]

def convert_to_object(data_string: str, format_type: str) -> Any:
    """
    Convert a data string in either JSON or XML format to the corresponding Python object.

    For JSON:
        - Returns the Python object represented by the JSON (dict, list, str, int, float, bool, or None).
    For XML:
        - Returns an xml.etree.ElementTree.Element representing the root of the parsed XML tree.

    Args:
        data_string (str): The input data as a string.
        format_type (str): The format of the input data, either 'json' or 'xml' (case-insensitive).

    Returns:
        Any: The Python object corresponding to the parsed data.

    Raises:
        TypeError: If data_string or format_type is not a string.
        ValueError: If the format_type is unsupported or the input data is invalid for the specified format.
    """
    if not isinstance(data_string, str):
        raise TypeError("data_string must be a string")
    if not isinstance(format_type, str):
        raise TypeError("format_type must be a string")

    fmt = format_type.strip().lower()

    if fmt == "json":
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(str(e)) from e
    elif fmt == "xml":
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(str(e)) from e
    else:
        raise ValueError("Unsupported format_type. Supported formats are 'json' and 'xml'.")

def load_serialized_data(bytes_data: bytes, data_format: str) -> Any:
    """
    Load serialized data from raw bytes according to the specified format and return the corresponding Python object.

    Supported formats are restricted to those listed in the global 'secure_formats'.
    By default, secure_formats = ["json", "xml"]. If a format (e.g., 'pickle') is not included
    in 'secure_formats', it is treated as unsupported/unsafe and will raise ValueError.

    For 'json':
        - Decodes bytes as UTF-8 and parses JSON into native Python types.
    For 'xml':
        - Decodes bytes as UTF-8 and parses into an xml.etree.ElementTree.Element.
    For 'pickle' (only if included in secure_formats):
        - Deserializes using pickle.loads without decoding.

    Args:
        bytes_data (bytes): The serialized data as raw bytes.
        data_format (str): The data format, e.g., 'json', 'xml', or 'pickle' (case-insensitive).

    Returns:
        Any: The deserialized Python object.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If the format is unsupported/unsafe or the data cannot be deserialized.
    """
    if not isinstance(bytes_data, (bytes, bytearray)):
        raise TypeError("bytes_data must be bytes or bytearray")
    if not isinstance(data_format, str):
        raise TypeError("data_format must be a string")

    fmt = data_format.strip().lower()
    allowed = {f.lower() for f in secure_formats}

    if fmt not in allowed:
        raise ValueError(f"Unsupported or unsafe data_format '{data_format}'. "
                         f"Allowed formats: {', '.join(sorted(allowed)) or '(none)'}")

    if fmt in ("json", "xml"):
        # Decode bytes as UTF-8 for text-based formats
        try:
            text = bytes_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode input bytes as UTF-8: {e}") from e

        if fmt == "json":
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(str(e)) from e
        else:  # fmt == "xml"
            try:
                return ET.fromstring(text)
            except ET.ParseError as e:
                raise ValueError(str(e)) from e

    if fmt == "pickle":
        # Only reachable if 'pickle' is included in secure_formats
        try:
            import pickle  # Imported here to avoid unnecessary exposure when not used
            data_bytes = bytes(bytes_data) if isinstance(bytes_data, bytearray) else bytes_data
            return pickle.loads(data_bytes)
        except Exception as e:
            raise ValueError(str(e)) from e

    # Defensive fallback (should not be reached due to checks above)
    raise ValueError(f"Unsupported data_format '{data_format}'.")
