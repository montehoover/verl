import json
import xml.etree.ElementTree as ET
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deserialization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

approved_formats = ["json", "xml"]

def validate_format(format_name):
    """Validate if the format is approved and safe."""
    if format_name not in approved_formats:
        logger.error(f"Validation failed: Unsupported or dangerous format '{format_name}'")
        raise ValueError(f"Unsupported or dangerous format: {format_name}")
    logger.debug(f"Format '{format_name}' validated successfully")

def deserialize_json(data_bytes):
    """Deserialize JSON bytes to Python object."""
    try:
        result = json.loads(data_bytes.decode('utf-8'))
        logger.debug("JSON deserialization successful")
        return result
    except Exception as e:
        logger.error(f"JSON deserialization failed: {str(e)}")
        raise

def deserialize_xml(data_bytes):
    """Deserialize XML bytes to Python object."""
    try:
        root = ET.fromstring(data_bytes.decode('utf-8'))
        result = _xml_to_dict(root)
        logger.debug("XML deserialization successful")
        return result
    except Exception as e:
        logger.error(f"XML deserialization failed: {str(e)}")
        raise

def get_deserializer(format_name):
    """Get the appropriate deserializer function for the format."""
    deserializers = {
        "json": deserialize_json,
        "xml": deserialize_xml
    }
    return deserializers.get(format_name)

def bytes_to_obj(data_bytes, format_name):
    logger.info(f"Attempting deserialization with format: '{format_name}'")
    try:
        validate_format(format_name)
        deserializer = get_deserializer(format_name)
        result = deserializer(data_bytes)
        logger.info(f"Deserialization successful for format: '{format_name}'")
        return result
    except Exception as e:
        logger.error(f"Deserialization failed for format '{format_name}': {str(e)}")
        raise

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
