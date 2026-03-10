import json
import xml.etree.ElementTree as ET

def convert_to_object(data_string, format_type):
    if format_type == 'json':
        return json.loads(data_string)
    elif format_type == 'xml':
        root = ET.fromstring(data_string)
        return element_to_dict(root)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def element_to_dict(element):
    result = {}
    
    # Add attributes
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Add text content
    if element.text and element.text.strip():
        if len(element) == 0:  # No children
            if element.attrib:
                result['#text'] = element.text.strip()
            else:
                return element.text.strip()
        else:
            result['#text'] = element.text.strip()
    
    # Add children
    for child in element:
        child_data = element_to_dict(child)
        if child.tag in result:
            # Convert to list if multiple children with same tag
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    
    return result
