import json
import xml.etree.ElementTree as ET

allowed_serialization_types = ["json", "xml"]

def deserialize_data(raw_data, serialization_format):
    if serialization_format not in allowed_serialization_types:
        raise ValueError(f"Unsupported format: {serialization_format}")
    
    if serialization_format == 'json':
        return json.loads(raw_data.decode('utf-8'))
    elif serialization_format == 'xml':
        root = ET.fromstring(raw_data.decode('utf-8'))
        return element_to_dict(root)

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
