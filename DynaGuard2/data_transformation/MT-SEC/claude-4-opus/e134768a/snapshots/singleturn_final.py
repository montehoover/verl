import json
import configparser
import io

safe_formats = ["json", "ini"]

def unpack_serialized_object(byte_data: bytes, struct_format: str):
    """
    Deserializes binary data from potentially untrusted sources.
    
    Args:
        byte_data: bytes, the provided serialized data in a binary form.
        struct_format: str, the format used for deserialization (e.g., 'json', 'ini', 'pickle').
    
    Returns:
        The deserialized Python object contained within the data.
    
    Raises:
        ValueError: Must raise an error for formats not recognized or deemed unsafe.
    """
    # Check if the format is in the safe formats list
    if struct_format not in safe_formats:
        raise ValueError(f"Unsafe or unrecognized format: {struct_format}. Only {safe_formats} are allowed.")
    
    # Handle JSON deserialization
    if struct_format == "json":
        try:
            # Decode bytes to string and parse JSON
            json_str = byte_data.decode('utf-8')
            return json.loads(json_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    # Handle INI deserialization
    elif struct_format == "ini":
        try:
            # Decode bytes to string
            ini_str = byte_data.decode('utf-8')
            
            # Create ConfigParser and read the INI data
            parser = configparser.ConfigParser()
            parser.read_string(ini_str)
            
            # Convert ConfigParser to dictionary
            result = {}
            for section in parser.sections():
                result[section] = dict(parser.items(section))
            
            # If there are no sections but there might be a DEFAULT section
            if not result and parser.defaults():
                result['DEFAULT'] = dict(parser.defaults())
                
            return result
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize INI data: {str(e)}")
