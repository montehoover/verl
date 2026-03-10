import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union


def convert_string_to_data(data_string: str, data_format: str) -> Union[Dict[str, Any], ET.Element]:
    """
    Convert an input string into a structured data object based on the specified format.

    Args:
        data_string (str): The input data as a string.
        data_format (str): The format of the input data ('json' or 'xml').

    Returns:
        Union[dict, xml.etree.ElementTree.Element]: A dictionary for JSON inputs or
        the root XML Element for XML inputs.

    Raises:
        ValueError: If the input is not valid JSON/XML, does not represent a JSON object,
        or if the format is unsupported.
    """
    if not isinstance(data_format, str):
        raise ValueError("Unsupported format: data_format must be 'json' or 'xml'")

    fmt = data_format.strip().lower()

    if fmt == "json":
        try:
            parsed = json.loads(data_string)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid JSON") from exc

        if not isinstance(parsed, dict):
            raise ValueError("JSON does not represent an object")

        return parsed

    if fmt == "xml":
        try:
            element = ET.fromstring(data_string)
        except (TypeError, ET.ParseError) as exc:
            raise ValueError("Invalid XML") from exc
        return element

    raise ValueError(f"Unsupported format: {data_format}")


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Backwards-compatible helper to convert JSON string to dict.
    """
    result = convert_string_to_data(json_string, "json")
    # At runtime this will be a dict due to validation in convert_string_to_data
    return result  # type: ignore[return-value]
