import json
from typing import Any, Dict, Union
from xml.etree.ElementTree import Element, fromstring, ParseError

allowed_formats = ["json", "html"]


def deserialize_stream_payload(serialized_data: bytes, data_format_type: str) -> Any:
    """
    Deserialize a byte stream from potentially untrusted sources based on the specified format.

    Args:
        serialized_data: The input byte stream.
        data_format_type: The format of the serialized data ('json', 'html', 'pickle').

    Returns:
        A Python object representing the deserialized data.
        - For 'json': the parsed JSON object (dict, list, etc.).
        - For 'html': a decoded UTF-8 string containing the HTML.

    Raises:
        ValueError: If the format is unsupported/insecure or the data cannot be parsed/decoded.
    """
    fmt = (data_format_type or "").strip().lower()

    if fmt not in allowed_formats:
        raise ValueError(
            f"Unsupported or insecure format: {data_format_type!r}. "
            f"Allowed formats: {', '.join(allowed_formats)}"
        )

    if fmt == "json":
        try:
            text = serialized_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 data for JSON: {e}") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

    if fmt == "html":
        try:
            return serialized_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 data for HTML: {e}") from e

    # Should not reach here because of allowed_formats check
    raise ValueError(f"Unsupported format: {data_format_type!r}")


def convert_to_dict(content_string: str, format_type: str) -> Union[Dict[str, Any], Element]:
    """
    Convert the given content string into a structured Python object based on the specified format.

    - For 'json': returns a dictionary parsed from the JSON string.
    - For 'xml': returns an xml.etree.ElementTree.Element parsed from the XML string.

    Raises:
        ValueError: If the format is unsupported or the content is invalid for the given format.
    """
    fmt = (format_type or "").strip().lower()

    if fmt == "json":
        try:
            data = json.loads(content_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

        if not isinstance(data, dict):
            raise ValueError("JSON must represent an object at the top level")

        return data

    if fmt == "xml":
        try:
            element = fromstring(content_string)
        except ParseError as e:
            raise ValueError(f"Invalid XML: {e}") from e
        return element

    raise ValueError(f"Unsupported format_type: {format_type!r}. Supported formats are 'json' and 'xml'.")


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Backwards-compatible helper that parses JSON strings into dictionaries.
    """
    result = convert_to_dict(json_string, "json")
    # mypy/type-checkers: result is Dict[str, Any] when format_type == 'json'
    return result  # type: ignore[return-value]
