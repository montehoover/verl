import json
import xml.etree.ElementTree as ET
import logging

class SizeLimitException(Exception):
    pass

def load_serialized_content(serialized_input, serialization_format):
    MAX_SIZE = 1024 * 1024  # 1MB limit
    
    if len(serialized_input) > MAX_SIZE:
        raise SizeLimitException(f"Input size {len(serialized_input)} exceeds maximum allowed size {MAX_SIZE}")
    
    try:
        # Decode bytes to string
        text_data = serialized_input.decode('utf-8')
        
        if serialization_format.lower() == 'json':
            logging.info("Attempting to parse as JSON")
            return json.loads(text_data)
        elif serialization_format.lower() == 'xml':
            logging.info("Attempting to parse as XML")
            return ET.fromstring(text_data)
        else:
            logging.info(f"Unknown format '{serialization_format}', returning as plain text")
            return text_data
            
    except (json.JSONDecodeError, ET.ParseError, UnicodeDecodeError) as e:
        logging.error(f"Failed to parse {serialization_format}: {str(e)}")
        return "Error: Failed to parse data"
