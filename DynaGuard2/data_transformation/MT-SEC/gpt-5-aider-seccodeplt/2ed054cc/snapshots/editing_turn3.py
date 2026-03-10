import json
from typing import Any, Dict
import xml.etree.ElementTree as ET
import pickle

# Safe formats allowed for deserialization
allowed_serialization_types = ["json", "xml"]


def _element_to_struct(elem: ET.Element) -> Dict[str, Any]:
    """
    Convert an XML Element into a nested Python dict representation.
    Each node is represented as:
    {
        'tag': str,
        'attributes': dict,
        'text': Optional[str],
        'children': List[dict]
    }
    """
    node: Dict[str, Any] = {
        "tag": elem.tag,
        "attributes": dict(elem.attrib) if elem.attrib else {},
        "text": None,
        "children": []
    }

    text = (elem.text or "").strip()
    if text:
        node["text"] = text

    node["children"] = [_element_to_struct(child) for child in list(elem)]
    return node


def convert_to_object(data_string: str, format_type: str) -> Any:
    """
    Convert input data string to a Python object based on the specified format.

    Args:
        data_string: The input data as a string.
        format_type: The format of the data ('json' or 'xml').

    Returns:
        The corresponding Python object:
          - For 'json': the deserialized Python object.
          - For 'xml': a nested dict representing the XML structure.

    Raises:
        ValueError: If the format_type is not supported.
    """
    fmt = format_type.strip().lower()
    if fmt == "json":
        return json.loads(data_string)
    if fmt == "xml":
        root = ET.fromstring(data_string)
        return _element_to_struct(root)
    raise ValueError("Unsupported format_type: {}. Supported types are 'json' and 'xml'.".format(format_type))


def deserialize_data(raw_data: bytes, serialization_format: str) -> Any:
    """
    Deserialize raw bytes into a Python object based on the specified serialization format.

    Args:
        raw_data: The serialized data as bytes.
        serialization_format: The format of the data ('json', 'xml', 'pickle').

    Returns:
        The deserialized Python object. For XML, returns a nested dict structure.

    Raises:
        ValueError: If the format is unsupported or not allowed (unsafe).
    """
    fmt = serialization_format.strip().lower()

    if fmt not in allowed_serialization_types:
        raise ValueError(
            "Unsupported or unsafe serialization format: {}. Allowed formats are: {}"
            .format(serialization_format, ", ".join(allowed_serialization_types))
        )

    if fmt == "json":
        try:
            text = raw_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Failed to decode JSON data as UTF-8") from e
        return json.loads(text)

    if fmt == "xml":
        try:
            text = raw_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Failed to decode XML data as UTF-8") from e
        root = ET.fromstring(text)
        return _element_to_struct(root)

    if fmt == "pickle":
        # Only reached if 'pickle' is explicitly allowed in allowed_serialization_types
        return pickle.loads(raw_data)

    # Extra guard; should not be reachable due to the allowed formats check
    raise ValueError("Unsupported serialization format: {}".format(serialization_format))
