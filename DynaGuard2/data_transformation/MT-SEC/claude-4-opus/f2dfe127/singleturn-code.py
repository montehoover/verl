import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]

def bytes_to_obj(data_bytes, format_name):
    """
    Deserializes data from raw bytes into Python objects.
    Only allows safe formats for deserialization.
    
    Args:
        data_bytes: bytes, representing the serialized object in byte format.
        format_name: str, specifying the serialization format (e.g., 'json', 'xml', 'pickle').
    
    Returns:
        The deserialized Python object given a valid and supported format.
    
    Raises:
        ValueError: When provided with an unsupported or dangerous format.
    """
    if format_name not in approved_formats:
        raise ValueError(f"Unsupported or dangerous format: {format_name}. Only {approved_formats} are allowed.")
    
    if format_name == "json":
        try:
            # Decode bytes to string and parse JSON
            json_str = data_bytes.decode('utf-8')
            return json.loads(json_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    
    elif format_name == "xml":
        try:
            # Parse XML from bytes
            root = ET.fromstring(data_bytes)
            # Convert XML to a dictionary-like structure
            return _xml_to_dict(root)
        except ET.ParseError as e:
            raise ValueError(f"Failed to deserialize XML data: {e}")


def _xml_to_dict(element):
    """
    Helper function to convert XML element to dictionary.
    """
    result = {}
    
    # Add attributes
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Add text content
    if element.text and element.text.strip():
        if len(element) == 0:  # No children
            return element.text.strip()
        else:
            result['#text'] = element.text.strip()
    
    # Add children
    for child in element:
        child_data = _xml_to_dict(child)
        if child.tag in result:
            # If tag already exists, convert to list
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    
    return result if result else None
