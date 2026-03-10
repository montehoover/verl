import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]

def validate_format(format_name):
    """Validate if the format is approved and safe."""
    if format_name not in approved_formats:
        raise ValueError(f"Unsupported or dangerous format: {format_name}")

def deserialize_json(data_bytes):
    """Deserialize JSON bytes to Python object."""
    return json.loads(data_bytes.decode('utf-8'))

def deserialize_xml(data_bytes):
    """Deserialize XML bytes to Python object."""
    root = ET.fromstring(data_bytes.decode('utf-8'))
    return _xml_to_dict(root)

def get_deserializer(format_name):
    """Get the appropriate deserializer function for the format."""
    deserializers = {
        "json": deserialize_json,
        "xml": deserialize_xml
    }
    return deserializers.get(format_name)

def bytes_to_obj(data_bytes, format_name):
    validate_format(format_name)
    deserializer = get_deserializer(format_name)
    return deserializer(data_bytes)

def _xml_to_dict(element):
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
    for child in element:
        child_data = _xml_to_dict(child)
        if child.tag in result:
            # Convert to list if multiple children with same tag
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    
    return result
