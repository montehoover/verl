import json
import xml.etree.ElementTree as ET

safe_formats = ["json", "xml"]

def deserialize_data(input_data, data_type):
    """
    Deserializes data from raw bytes into Python objects.
    
    Args:
        input_data: bytes, representing the serialized object in byte format.
        data_type: str, specifying the serialization format (e.g., 'json', 'xml', 'pickle').
    
    Returns:
        The deserialized Python object given a valid and supported format.
    
    Raises:
        ValueError: When provided with an unsupported or dangerous format.
    """
    # Check if the data type is in the safe formats list
    if data_type not in safe_formats:
        raise ValueError(f"Unsupported or dangerous format: {data_type}. Only {safe_formats} are allowed.")
    
    # Handle JSON deserialization
    if data_type == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = input_data.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    # Handle XML deserialization
    elif data_type == "xml":
        try:
            # Parse XML from bytes
            root = ET.fromstring(input_data.decode('utf-8'))
            # Convert XML to a dictionary representation
            return _xml_to_dict(root)
        except (ET.ParseError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize XML data: {str(e)}")


def _xml_to_dict(element):
    """
    Helper function to convert XML element to dictionary.
    """
    result = {}
    
    # Handle attributes
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Handle text content
    if element.text and element.text.strip():
        if len(element) == 0:  # No children
            return element.text.strip()
        else:
            result['text'] = element.text.strip()
    
    # Handle children
    children = {}
    for child in element:
        child_data = _xml_to_dict(child)
        if child.tag in children:
            # If tag already exists, convert to list
            if not isinstance(children[child.tag], list):
                children[child.tag] = [children[child.tag]]
            children[child.tag].append(child_data)
        else:
            children[child.tag] = child_data
    
    if children:
        result.update(children)
    
    # If only has attributes or is empty, return the result
    # Otherwise, if it's just text, return the text
    if not result:
        return None
    elif len(result) == 1 and 'text' not in result and '@attributes' not in result:
        return list(result.values())[0]
    else:
        return result
