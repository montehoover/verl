import json
import xml.etree.ElementTree as ET

secure_formats = ["json", "xml"]

def load_serialized_data(bytes_data: bytes, data_format: str):
    if data_format not in secure_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")
    
    if data_format == "json":
        try:
            return json.loads(bytes_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    
    elif data_format == "xml":
        try:
            root = ET.fromstring(bytes_data.decode('utf-8'))
            return _xml_to_dict(root)
        except (ET.ParseError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize XML data: {e}")

def _xml_to_dict(element):
    result = {}
    
    # Handle attributes
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Handle text content
    if element.text and element.text.strip():
        text = element.text.strip()
        if len(element) == 0 and not element.attrib:
            return text
        else:
            result['#text'] = text
    
    # Handle child elements
    for child in element:
        child_data = _xml_to_dict(child)
        if child.tag in result:
            # If the tag already exists, convert to list
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    
    return result if result else None
