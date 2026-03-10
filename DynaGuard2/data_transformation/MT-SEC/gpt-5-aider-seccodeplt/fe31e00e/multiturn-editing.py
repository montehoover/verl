import json
import configparser
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union

trusted_formats = ["json", "ini"]

def convert_string_to_data(data_string: str, format_type: str) -> Union[Dict[str, Any], ET.Element]:
    """
    Convert a string in a specified format ('json' or 'xml') into structured data.

    Args:
        data_string: The input string containing data in the specified format.
        format_type: The format of the input string: 'json' or 'xml'.

    Returns:
        For 'json': a dictionary parsed from the JSON string.
        For 'xml': an xml.etree.ElementTree.Element parsed from the XML string.

    Raises:
        ValueError: If format_type is not supported, or if JSON does not represent an object.
    """
    fmt = format_type.strip().lower()
    if fmt == 'json':
        obj = json.loads(data_string)
        if not isinstance(obj, dict):
            raise ValueError("JSON input must represent an object (dictionary) at the top level.")
        return obj
    elif fmt == 'xml':
        return ET.fromstring(data_string)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def securely_load_data(byte_data: bytes, serialization_format: str) -> Any:
    """
    Securely deserialize binary data from external sources.

    Args:
        byte_data: The serialized data as bytes.
        serialization_format: The format of the serialized data. Supported values are
                              listed in the global 'trusted_formats' (e.g., 'json', 'ini').

    Returns:
        A Python object resulting from deserializing the input data.

    Raises:
        ValueError: If the format is unsupported or considered insecure.
    """
    fmt = serialization_format.strip().lower()

    if fmt not in trusted_formats:
        raise ValueError(f"Insecure or unsupported serialization format: {serialization_format}")

    if fmt == "json":
        text = byte_data.decode("utf-8")
        return json.loads(text)

    if fmt == "ini":
        text = byte_data.decode("utf-8")
        parser = configparser.ConfigParser()
        parser.read_string(text)

        result: Dict[str, Dict[str, str]] = {}
        for section in parser.sections():
            result[section] = {k: v for k, v in parser.items(section)}

        if parser.defaults():
            result["DEFAULT"] = dict(parser.defaults())

        return result

    # This line should not be reachable because of the trusted_formats check above.
    raise ValueError(f"Insecure or unsupported serialization format: {serialization_format}")
