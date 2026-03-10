import json
import xml.etree.ElementTree as ET

secure_formats = ["json", "xml"]

def load_serialized_data(bytes_data, data_format):
    """
    Safely converts serialized data from bytes into Python objects.
    
    Args:
        bytes_data: bytes, representing the serialized form of the object
        data_format: str, indicating the data format used for serialization
        
    Returns:
        The deserialized Python data structure
        
    Raises:
        ValueError: When an unsupported or unsafe format is provided
    """
    # Check if the format is in the secure formats list
    if data_format not in secure_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")
    
    # Handle JSON deserialization
    if data_format == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = bytes_data.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    # Handle XML deserialization
    elif data_format == "xml":
        try:
            # Parse XML from bytes
            root = ET.fromstring(bytes_data)
            # Convert XML to a dictionary-like structure
            return _xml_to_dict(root)
        except ET.ParseError as e:
            raise ValueError(f"Failed to deserialize XML data: {str(e)}")
    
    # This should not be reached due to the initial check, but included for safety
    raise ValueError(f"Unsupported format: {data_format}")


def _xml_to_dict(element):
    """
    Helper function to convert XML element to dictionary.
    """
    result = {}
    
    # Add attributes if any
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Process child elements
    children = list(element)
    if children:
        child_dict = {}
        for child in children:
            child_data = _xml_to_dict(child)
            if child.tag in child_dict:
                # If tag already exists, make it a list
                if not isinstance(child_dict[child.tag], list):
                    child_dict[child.tag] = [child_dict[child.tag]]
                child_dict[child.tag].append(child_data)
            else:
                child_dict[child.tag] = child_data
        result.update(child_dict)
    
    # Add text content if any
    if element.text and element.text.strip():
        if children or element.attrib:
            result['#text'] = element.text.strip()
        else:
            return element.text.strip()
    
    return result or None
