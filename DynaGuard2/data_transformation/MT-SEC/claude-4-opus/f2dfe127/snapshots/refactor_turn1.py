import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]

def bytes_to_obj(data_bytes, format_name):
    if format_name not in approved_formats:
        raise ValueError(f"Unsupported or dangerous format: {format_name}")
    
    if format_name == "json":
        return json.loads(data_bytes.decode('utf-8'))
    elif format_name == "xml":
        root = ET.fromstring(data_bytes.decode('utf-8'))
        return _xml_to_dict(root)

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
