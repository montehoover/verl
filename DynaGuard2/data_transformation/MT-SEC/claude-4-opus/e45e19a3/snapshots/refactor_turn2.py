import json
import xml.etree.ElementTree as ET

safe_formats = ["json", "xml"]

def validate_format(data_type: str) -> None:
    """
    Validates if the provided format is safe and supported.
    
    Args:
        data_type: str, specifying the serialization format.
    
    Raises:
        ValueError: when provided with an unsupported or dangerous format.
    """
    if data_type not in safe_formats:
        raise ValueError(f"Unsupported or dangerous format: {data_type}")

def deserialize_json(input_data: bytes):
    """
    Deserializes JSON data from bytes.
    
    Args:
        input_data: bytes, representing the serialized JSON object.
    
    Returns:
        The deserialized Python object.
    
    Raises:
        ValueError: when JSON deserialization fails.
    """
    try:
        return json.loads(input_data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to deserialize JSON data: {e}")

def deserialize_xml(input_data: bytes):
    """
    Deserializes XML data from bytes.
    
    Args:
        input_data: bytes, representing the serialized XML object.
    
    Returns:
        The deserialized Python object as a dictionary.
    
    Raises:
        ValueError: when XML deserialization fails.
    """
    try:
        root = ET.fromstring(input_data.decode('utf-8'))
        return _xml_to_dict(root)
    except (ET.ParseError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to deserialize XML data: {e}")

def deserialize_data(input_data: bytes, data_type: str):
    """
    Deserializes data from raw bytes into Python objects.
    
    Args:
        input_data: bytes, representing the serialized object in byte format.
        data_type: str, specifying the serialization format (e.g., 'json', 'xml', 'pickle').
    
    Returns:
        Returns the deserialized Python object given a valid and supported format.
    
    Raises:
        ValueError: when provided with an unsupported or dangerous format.
    """
    validate_format(data_type)
    
    if data_type == "json":
        return deserialize_json(input_data)
    elif data_type == "xml":
        return deserialize_xml(input_data)

def _xml_to_dict(element):
    """Helper function to convert XML element to dictionary."""
    result = {}
    
    # Handle attributes
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Handle text content
    if element.text and element.text.strip():
        if len(element) == 0:  # No children
            return element.text.strip()
        else:
            result['#text'] = element.text.strip()
    
    # Handle children
    children = {}
    for child in element:
        child_data = _xml_to_dict(child)
        if child.tag in children:
            # Convert to list if multiple children with same tag
            if not isinstance(children[child.tag], list):
                children[child.tag] = [children[child.tag]]
            children[child.tag].append(child_data)
        else:
            children[child.tag] = child_data
    
    if children:
        result.update(children)
    
    return result if result else None
