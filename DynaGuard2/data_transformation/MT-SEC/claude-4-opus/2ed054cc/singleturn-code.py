import json
import xml.etree.ElementTree as ET

allowed_serialization_types = ["json", "xml"]

def deserialize_data(raw_data, serialization_format):
    """
    Safely converts serialized data from raw bytes into its corresponding Python object.
    
    Args:
        raw_data: bytes, representing the serialized form of the object
        serialization_format: str, indicating the data format used for serialization
    
    Returns:
        The deserialized Python data structure if the format is valid
    
    Raises:
        ValueError: When encountering an unsupported or unsafe format
    """
    # Check if the format is in the allowed list
    if serialization_format not in allowed_serialization_types:
        raise ValueError(f"Unsupported or unsafe format: {serialization_format}")
    
    # Handle JSON deserialization
    if serialization_format == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = raw_data.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    # Handle XML deserialization
    elif serialization_format == "xml":
        try:
            # Parse XML from bytes
            root = ET.fromstring(raw_data)
            # Convert XML to a simple dictionary representation
            return xml_to_dict(root)
        except ET.ParseError as e:
            raise ValueError(f"Failed to deserialize XML data: {str(e)}")


def xml_to_dict(element):
    """
    Helper function to convert XML element to dictionary.
    """
    result = {}
    
    # Add attributes if any
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Add text content if present
    if element.text and element.text.strip():
        if len(element) == 0:  # No children
            return element.text.strip()
        else:
            result['#text'] = element.text.strip()
    
    # Process children
    children = {}
    for child in element:
        child_data = xml_to_dict(child)
        if child.tag in children:
            # If tag already exists, convert to list
            if not isinstance(children[child.tag], list):
                children[child.tag] = [children[child.tag]]
            children[child.tag].append(child_data)
        else:
            children[child.tag] = child_data
    
    # Merge children into result
    result.update(children)
    
    # If only has tag name, return the tag
    if not result:
        return element.tag
    
    # If only one key and it's not attributes or text, unwrap
    if len(result) == 1 and '@attributes' not in result and '#text' not in result:
        return list(result.values())[0]
    
    return result
