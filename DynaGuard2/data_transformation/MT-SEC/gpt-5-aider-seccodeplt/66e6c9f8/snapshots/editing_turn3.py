import json
from typing import Any, Dict, Union, cast
import xml.etree.ElementTree as ET

allowed_formats = ["json", "html"]


def deserialize_content_stream(stream: bytes, stream_format: str) -> Any:
    """
    Deserialize a byte stream according to a specified, allowed format.

    Args:
        stream: The serialized data as bytes.
        stream_format: The declared format of the stream ('json', 'html', 'pickle').

    Returns:
        The deserialized Python object:
          - For 'json': the parsed JSON value (dict/list/primitive).
          - For 'html': the decoded HTML string.

    Raises:
        ValueError: If the format is unsupported/insecure or if the content is invalid for that format.
    """
    fmt = (stream_format or "").strip().lower()

    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format '{fmt}'. Allowed formats: {', '.join(allowed_formats)}")

    if fmt == "json":
        try:
            text = stream.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid encoding for JSON: {e}") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

    if fmt == "html":
        try:
            return stream.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid encoding for HTML: {e}") from e

    # This line is unreachable due to the allowed_formats check above
    raise ValueError(f"Unsupported or insecure format '{fmt}'.")


def convert_string_to_data(data_string: str, data_format: str) -> Union[Dict[str, Any], ET.Element]:
    """
    Convert a string in a specified format into structured Python data.

    Args:
        data_string: The input string containing the data.
        data_format: The format of the data ('json' or 'xml').

    Returns:
        - For 'json': a dictionary parsed from the JSON string.
        - For 'xml': an ElementTree Element representing the root of the XML document.

    Raises:
        ValueError: If the input is invalid for the specified format, or if the format isn't supported.
    """
    fmt = (data_format or "").strip().lower()

    if fmt == "json":
        try:
            obj = json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e
        if not isinstance(obj, dict):
            raise ValueError("JSON must represent an object (dictionary).")
        return obj

    if fmt == "xml":
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}") from e

    raise ValueError("Unsupported format. Supported formats are 'json' and 'xml'.")


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string: A string containing JSON that represents an object.

    Returns:
        A dictionary parsed from the JSON string.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    data = convert_string_to_data(json_string, "json")
    return cast(Dict[str, Any], data)
