import json
import configparser
import io

trusted_formats = ["json", "ini"]

def decode_serialized_data(data_bytes, format_string):
    """
    Safely deserialize data based on defined trusted formats.
    
    Args:
        data_bytes: bytes, the incoming serialized data in binary format.
        format_string: str, a string identifying the format of serialization (e.g. 'json', 'ini', 'pickle').
    
    Returns:
        A Python object resulted from deserializing the data.
    
    Raises:
        ValueError: Raises exceptions for unsupported or untrusted formats.
    """
    # Check if the format is in the trusted formats list
    if format_string not in trusted_formats:
        raise ValueError(f"Untrusted or unsupported format: {format_string}")
    
    # Handle JSON format
    if format_string == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = data_bytes.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode JSON data: {str(e)}")
    
    # Handle INI format
    elif format_string == "ini":
        try:
            # Decode bytes to string
            ini_string = data_bytes.decode('utf-8')
            
            # Create a ConfigParser object
            config = configparser.ConfigParser()
            
            # Read the INI data from a string buffer
            config.read_string(ini_string)
            
            # Convert ConfigParser to a dictionary
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            
            # If there are no sections but there might be a DEFAULT section
            if not result and config.defaults():
                result['DEFAULT'] = dict(config.defaults())
            
            return result
            
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode INI data: {str(e)}")
    
    # This should not be reached due to the initial check, but included for safety
    else:
        raise ValueError(f"Format {format_string} is not implemented")
