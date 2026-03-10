import json
from typing import Dict, Any, Union
from xml.etree.ElementTree import fromstring, Element, ParseError

# Only safe, predefined formats are allowed for deserialization.
allowed_formats = ["json", "html"]


def convert_to_dict(data_string: str, data_format: str) -> Union[Dict[str, Any], Element]:
    """
    Convert a data string into a structured Python object based on the given format.

    For JSON:
      - Parses the string and returns a dictionary.
      - Raises ValueError if the JSON is invalid or does not represent an object.

    For XML:
      - Parses the string and returns an xml.etree.ElementTree.Element.
      - Raises ValueError if the XML is invalid.

    Args:
        data_string: The input data as a string.
        data_format: The format of the input data ('json' or 'xml').

    Returns:
        A dictionary for JSON input, or an Element for XML input.

    Raises:
        ValueError: If arguments are invalid, parsing fails, or the format is unsupported.
    """
    if not isinstance(data_string, str):
        raise ValueError("data_string must be a string")
    if not isinstance(data_format, str):
        raise ValueError("data_format must be a string specifying 'json' or 'xml'")

    fmt = data_format.strip().lower()

    if fmt == 'json':
        try:
            data = json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e
        if not isinstance(data, dict):
            raise ValueError("JSON must represent an object")
        return data

    if fmt == 'xml':
        try:
            element = fromstring(data_string)
        except ParseError as e:
            raise ValueError(f"Invalid XML: {e.msg}") from e
        return element

    raise ValueError("Unsupported format: must be 'json' or 'xml'")


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper to convert a JSON string into a Python dictionary.

    Args:
        json_string: A JSON-formatted string representing an object.

    Returns:
        A dictionary parsed from the JSON string.

    Raises:
        ValueError: If the input is not a string, not valid JSON, or does not represent a JSON object.
    """
    result = convert_to_dict(json_string, 'json')
    # Type guard for static checkers; runtime guaranteed by convert_to_dict.
    return result  # type: ignore[return-value]


def deserialize_stream(serialized_data: bytes, stream_format: str) -> Any:
    """
    Deserialize a byte stream into a Python object in a secure manner.

    Supported and safe formats are defined in the global `allowed_formats`.

    - 'json': Decodes bytes as UTF-8 and parses JSON into Python objects.
    - 'html': Decodes bytes as UTF-8 and returns the HTML text (no execution or parsing).

    Args:
        serialized_data: The data to deserialize, as bytes.
        stream_format: The format of the data ('json', 'html', or others).

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If inputs are invalid, the format is unsupported or insecure,
                    or deserialization fails.
    """
    if not isinstance(serialized_data, bytes):
        raise ValueError("serialized_data must be bytes")
    if not isinstance(stream_format, str):
        raise ValueError("stream_format must be a string")

    fmt = stream_format.strip().lower()

    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}")

    if fmt == "json":
        try:
            text = serialized_data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 encoding: {e.reason}") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

    if fmt == "html":
        try:
            # Return the HTML as a plain string; do not parse or execute.
            return serialized_data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 encoding: {e.reason}") from e

    # Fallback; should not be reached due to allowed_formats gate.
    raise ValueError(f"Unsupported or insecure format: {stream_format}")
