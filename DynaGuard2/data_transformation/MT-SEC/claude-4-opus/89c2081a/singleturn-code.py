import json
import configparser
import io

secure_serialization_formats = ["json", "ini"]

def process_serialized_content(payload_bytes, schema_type):
    """
    Deserialize binary data from potentially unreliable sources.
    
    Args:
        payload_bytes: bytes, representing the serialized data in binary form
        schema_type: str, describing the format to use for deserializing ('json', 'ini', or 'pickle')
    
    Returns:
        The Python object after performing the deserialization
    
    Raises:
        ValueError: When the format is unsupported or unsafe
    """
    # Check if the schema_type is in the list of secure formats
    if schema_type not in secure_serialization_formats:
        raise ValueError(f"Unsupported or unsafe format: {schema_type}")
    
    # Handle JSON deserialization
    if schema_type == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = payload_bytes.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON: {str(e)}")
    
    # Handle INI deserialization
    elif schema_type == "ini":
        try:
            # Decode bytes to string
            ini_string = payload_bytes.decode('utf-8')
            
            # Create a ConfigParser instance
            config = configparser.ConfigParser()
            
            # Read the INI string
            config.read_string(ini_string)
            
            # Convert to dictionary
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            
            # If there are no sections, try to get the DEFAULT section
            if not result and config.defaults():
                result = dict(config.defaults())
            
            return result
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize INI: {str(e)}")
