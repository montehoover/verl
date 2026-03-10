import json
from typing import Any, Dict
import xml.etree.ElementTree as ET


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
