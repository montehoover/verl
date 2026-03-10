import json
import xml.etree.ElementTree as ET
import logging

# Configure logger
logger = logging.getLogger(__name__)

secure_formats = ["json", "xml"]

def _validate_format(data_format: str) -> None:
    """Validate that the data format is in the list of secure formats."""
    if data_format not in secure_formats:
        logger.error(f"Validation failed: Unsupported or unsafe format '{data_format}'")
        raise ValueError(f"Unsupported or unsafe format: {data_format}")

def _deserialize_json(bytes_data: bytes) -> any:
    """Deserialize JSON data from bytes."""
    try:
        logger.debug(f"Attempting JSON deserialization of {len(bytes_data)} bytes")
        result = json.loads(bytes_data.decode('utf-8'))
        logger.info(f"Successfully deserialized JSON data (type: {type(result).__name__})")
        return result
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"JSON deserialization failed: {type(e).__name__}: {e}")
        raise ValueError(f"Failed to deserialize JSON data: {e}")

def _deserialize_xml(bytes_data: bytes) -> dict:
    """Deserialize XML data from bytes."""
    try:
        logger.debug(f"Attempting XML deserialization of {len(bytes_data)} bytes")
        root = ET.fromstring(bytes_data.decode('utf-8'))
        result = _xml_to_dict(root)
        logger.info(f"Successfully deserialized XML data (root tag: {root.tag})")
        return result
    except (ET.ParseError, UnicodeDecodeError) as e:
        logger.error(f"XML deserialization failed: {type(e).__name__}: {e}")
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

def load_serialized_data(bytes_data: bytes, data_format: str):
    logger.info(f"Deserialization attempt started for format: '{data_format}'")
    
    try:
        _validate_format(data_format)
        
        if data_format == "json":
            result = _deserialize_json(bytes_data)
        elif data_format == "xml":
            result = _deserialize_xml(bytes_data)
        
        logger.info(f"Deserialization completed successfully for format: '{data_format}'")
        return result
    except Exception as e:
        logger.error(f"Deserialization failed for format: '{data_format}' - {type(e).__name__}: {e}")
        raise
