import json
import configparser
from typing import Dict, Any, Union
from xml.etree.ElementTree import Element, fromstring as xml_fromstring, ParseError

# A curated list of formats that have been evaluated to ensure safe deserialization.
secure_serialization_formats = ["json", "ini"]


def convert_to_dict(data_string: str, data_format: str) -> Union[Dict[str, Any], Element]:
    """
    Convert a data string into a structured object based on the specified format.

    For 'json': returns a Python dictionary parsed from the JSON string.
    For 'xml': returns an xml.etree.ElementTree.Element parsed from the XML string.

    Args:
        data_string: The input data as a string.
        data_format: The format of the input data ('json' or 'xml').

    Returns:
        Dict[str, Any]: If data_format is 'json'.
        Element: If data_format is 'xml'.

    Raises:
        TypeError: If inputs are not strings.
        ValueError: If the JSON is invalid or not an object, if the XML is invalid,
                    or if the format is unsupported.
    """
    if not isinstance(data_string, str):
        raise TypeError("data_string must be a string")
    if not isinstance(data_format, str):
        raise TypeError("data_format must be a string")

    fmt = data_format.strip().lower()

    if fmt == "json":
        try:
            data = json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON") from e

        if not isinstance(data, dict):
            raise ValueError("JSON must represent an object")

        return data

    if fmt == "xml":
        try:
            element = xml_fromstring(data_string)
        except ParseError as e:
            raise ValueError("Invalid XML") from e
        return element

    raise ValueError("Unsupported format: {0}. Supported formats are 'json' and 'xml'.".format(data_format))


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string: A string containing a JSON object.

    Returns:
        A Python dictionary parsed from the JSON string.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
        TypeError: If json_string is not a string.
    """
    return convert_to_dict(json_string, "json")  # type: ignore[return-value]


def process_serialized_content(payload_bytes: bytes, schema_type: str) -> Any:
    """
    Deserialize serialized binary data according to the specified schema type.

    Supports the following schema types in principle: 'json', 'ini', 'pickle'.
    However, only formats listed in `secure_serialization_formats` are allowed.
    Currently allowed: 'json', 'ini'.

    Args:
        payload_bytes: The serialized data as bytes.
        schema_type: The declared format of the payload ('json', 'ini', or 'pickle').

    Returns:
        The deserialized Python object.

    Raises:
        TypeError: If arguments are of incorrect types.
        ValueError: If the format is unsupported, unsafe, or the payload is invalid.
    """
    if not isinstance(payload_bytes, bytes):
        raise TypeError("payload_bytes must be bytes")
    if not isinstance(schema_type, str):
        raise TypeError("schema_type must be a string")

    fmt = schema_type.strip().lower()
    supported_formats = {"json", "ini", "pickle"}

    if fmt not in supported_formats:
        raise ValueError(f"Unsupported format: {schema_type}")

    if fmt not in secure_serialization_formats:
        # Explicitly disallow formats that are not marked as secure (e.g., 'pickle').
        raise ValueError(f"Unsafe or disallowed format: {schema_type}")

    if fmt == "json":
        try:
            # json.loads accepts bytes and decodes as UTF-8 per spec.
            return json.loads(payload_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
            raise ValueError("Invalid JSON payload") from e

    if fmt == "ini":
        try:
            text = payload_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("INI payload is not valid UTF-8") from e

        parser = configparser.ConfigParser()
        try:
            parser.read_string(text)
        except configparser.Error as e:
            raise ValueError("Invalid INI payload") from e
        return parser

    # Should not be reachable due to earlier checks
    raise ValueError(f"Unsupported format: {schema_type}")
