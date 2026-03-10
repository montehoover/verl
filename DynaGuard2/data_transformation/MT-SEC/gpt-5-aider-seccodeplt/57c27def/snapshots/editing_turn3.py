import json
import xml.etree.ElementTree as ET
import configparser
from typing import Any, Dict, Union

valid_formats = ["json", "ini"]

def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string (str): A string containing a JSON object.

    Returns:
        Dict[str, Any]: The parsed dictionary.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON must represent an object")

    return data

def convert_to_dict(data_string: str, format_type: str) -> Union[Dict[str, Any], ET.Element]:
    """
    Convert a data string into a structured object based on the specified format.

    For JSON:
        - Parses the string and returns a Python dictionary.
        - Raises ValueError if the JSON is invalid or does not represent an object.

    For XML:
        - Parses the string and returns an xml.etree.ElementTree.Element.
        - Raises ValueError if the XML is invalid.

    Args:
        data_string (str): The input data string to parse.
        format_type (str): The format of the data ('json' or 'xml').

    Returns:
        Union[Dict[str, Any], ET.Element]: A dictionary for JSON, or an XML Element for XML.

    Raises:
        ValueError: If the format is unsupported or parsing fails.
    """
    fmt = format_type.strip().lower()
    if fmt == 'json':
        return convert_json_to_dict(data_string)
    if fmt == 'xml':
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}") from e
    raise ValueError(f"Unsupported format: {format_type}")

def convert_serialized_data(raw_bytes: bytes, format_hint: str) -> Any:
    """
    Deserialize binary data according to the provided format hint.

    Supports only formats listed in `valid_formats`. Unsafe formats like 'pickle'
    are explicitly rejected.

    Args:
        raw_bytes (bytes): The serialized data as bytes.
        format_hint (str): The format hint ('json', 'ini', 'pickle').

    Returns:
        Any: The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported/unsafe or if deserialization fails.
    """
    fmt = format_hint.strip().lower()
    if fmt not in valid_formats:
        if fmt == 'pickle':
            raise ValueError("Unsupported or unsafe format: pickle")
        raise ValueError(f"Unsupported format: {format_hint}")

    try:
        text = raw_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError("Invalid text encoding: expected UTF-8") from e

    if fmt == 'json':
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e

    if fmt == 'ini':
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except (configparser.Error, ValueError) as e:
            raise ValueError(f"Invalid INI: {e}") from e

        result: Dict[str, Dict[str, str]] = {}
        if parser.defaults():
            result["DEFAULT"] = dict(parser.defaults())
        for section in parser.sections():
            result[section] = {k: v for k, v in parser.items(section, raw=True)}
        return result

    raise ValueError(f"Unsupported format: {format_hint}")
