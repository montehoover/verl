import json
import configparser
import io

valid_formats = ["json", "ini"]

def convert_serialized_data(raw_bytes, format_hint):
    """
    Deserialize binary data using only safe formats.
    
    Args:
        raw_bytes: bytes, representing the serialized data in binary form
        format_hint: str, describing the format to use for deserializing
        
    Returns:
        The Python object after performing the deserialization
        
    Raises:
        ValueError: If the format is unsupported or unsafe
    """
    if format_hint not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: {format_hint}. Only {valid_formats} are allowed.")
    
    if format_hint == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = raw_bytes.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    
    elif format_hint == "ini":
        try:
            # Decode bytes to string and parse INI
            ini_string = raw_bytes.decode('utf-8')
            config = configparser.ConfigParser()
            config.read_string(ini_string)
            
            # Convert ConfigParser to dictionary
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            
            # If there's a DEFAULT section, include it
            if config.defaults():
                result['DEFAULT'] = dict(config.defaults())
                
            return result
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize INI data: {e}")
