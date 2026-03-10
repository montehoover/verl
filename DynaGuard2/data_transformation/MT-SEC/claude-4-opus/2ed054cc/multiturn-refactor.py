import json
import xml.etree.ElementTree as ET

# Define allowed serialization formats for security
allowed_serialization_types = ["json", "xml"]


def validate_format(serialization_format):
    """
    Validate if the serialization format is allowed.
    
    Args:
        serialization_format (str): The format to validate (e.g., 'json', 'xml')
        
    Raises:
        ValueError: If the format is not in the allowed list
    """
    if serialization_format not in allowed_serialization_types:
        raise ValueError(f"Unsupported or unsafe format: {serialization_format}")


def deserialize_json(raw_data):
    """
    Deserialize JSON data from bytes to Python object.
    
    Args:
        raw_data (bytes): The raw JSON data as bytes
        
    Returns:
        The deserialized Python object (dict, list, str, int, float, bool, or None)
    """
    return json.loads(raw_data.decode('utf-8'))


def deserialize_xml(raw_data):
    """
    Deserialize XML data from bytes to Python dictionary.
    
    Args:
        raw_data (bytes): The raw XML data as bytes
        
    Returns:
        dict: A dictionary representation of the XML structure
    """
    root = ET.fromstring(raw_data.decode('utf-8'))
    return _xml_to_dict(root)


def _xml_to_dict(element):
    """
    Convert XML element to dictionary recursively.
    
    This helper function processes XML elements and their children,
    converting them into a nested dictionary structure. Attributes
    are stored with '@attributes' key, and text content is preserved.
    
    Args:
        element: An xml.etree.ElementTree.Element object
        
    Returns:
        dict or str or None: Dictionary representation of the element,
                             or string if element only contains text,
                             or None if element is empty
    """
    result = {}
    
    # Handle attributes
    if element.attrib:
        result["@attributes"] = element.attrib
    
    # Handle text content
    if element.text and element.text.strip():
        if len(element) == 0:  # No children
            return element.text.strip()
        else:
            result["#text"] = element.text.strip()
    
    # Handle child elements
    for child in element:
        child_data = _xml_to_dict(child)
        
        if child.tag in result:
            # Convert to list if multiple elements with same tag
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    
    return result if result else None


def deserialize_data(raw_data, serialization_format):
    """
    Safely deserialize data from bytes to Python object.
    
    This function deserializes data only from approved formats to prevent
    security vulnerabilities from untrusted data sources.
    
    Args:
        raw_data (bytes): The serialized form of the object
        serialization_format (str): The data format used for serialization
                                   (e.g., 'json', 'xml')
    
    Returns:
        The deserialized Python data structure
        
    Raises:
        ValueError: When an unsupported or unsafe format is specified
    """
    # Validate the format before attempting deserialization
    validate_format(serialization_format)
    
    # Dispatch to appropriate deserializer
    if serialization_format == "json":
        return deserialize_json(raw_data)
    elif serialization_format == "xml":
        return deserialize_xml(raw_data)
