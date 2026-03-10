import json
import configparser
import logging

approved_formats = ["json", "ini"]

class SizeLimitException(Exception):
    pass

class UnsafeFormatException(Exception):
    pass

class ParseException(Exception):
    pass

def load_serialized_content(serialized_input, serialization_format):
    MAX_SIZE = 1024 * 1024  # 1MB limit
    
    if len(serialized_input) > MAX_SIZE:
        raise SizeLimitException(f"Input size {len(serialized_input)} exceeds maximum allowed size {MAX_SIZE}")
    
    if serialization_format.lower() not in approved_formats:
        logging.error(f"Unsafe format attempted: {serialization_format}")
        raise UnsafeFormatException(f"Format '{serialization_format}' is not in approved formats: {approved_formats}")
    
    try:
        # Decode bytes to string
        text_data = serialized_input.decode('utf-8')
        
        if serialization_format.lower() == 'json':
            logging.info("Attempting to parse as JSON")
            parsed_data = json.loads(text_data)
            logging.info("Successfully parsed JSON data")
            return parsed_data
        elif serialization_format.lower() == 'ini':
            logging.info("Attempting to parse as INI")
            config = configparser.ConfigParser()
            config.read_string(text_data)
            # Convert to dictionary
            parsed_data = {section: dict(config.items(section)) for section in config.sections()}
            logging.info("Successfully parsed INI data")
            return parsed_data
            
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {str(e)}")
        raise ParseException(f"Failed to parse JSON: {str(e)}")
    except configparser.Error as e:
        logging.error(f"Failed to parse INI: {str(e)}")
        raise ParseException(f"Failed to parse INI: {str(e)}")
    except UnicodeDecodeError as e:
        logging.error(f"Failed to decode bytes to UTF-8: {str(e)}")
        raise ParseException(f"Failed to decode bytes to UTF-8: {str(e)}")
