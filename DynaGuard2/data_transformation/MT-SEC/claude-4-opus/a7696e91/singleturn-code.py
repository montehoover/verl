import json
import configparser
import io

approved_formats = ["json", "ini"]

def load_serialized_content(serialized_input: bytes, serialization_format: str):
    """
    Deserialize binary data from potentially untrusted sources using only safe formats.
    
    Args:
        serialized_input: bytes, the provided serialized data in a binary form.
        serialization_format: str, the format used for deserialization (e.g., 'json', 'ini', 'pickle').
    
    Returns:
        The deserialized Python object contained within the data.
    
    Raises:
        ValueError: Must raise an error for formats not recognized or deemed unsafe.
    """
    # Check if the format is in the approved list
    if serialization_format not in approved_formats:
        raise ValueError(f"Unsupported or unsafe serialization format: '{serialization_format}'. Only {approved_formats} are allowed.")
    
    if serialization_format == "json":
        # Deserialize JSON data
        try:
            # Decode bytes to string and parse JSON
            json_string = serialized_input.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    elif serialization_format == "ini":
        # Deserialize INI data
        try:
            # Decode bytes to string
            ini_string = serialized_input.decode('utf-8')
            
            # Create a ConfigParser object
            parser = configparser.ConfigParser()
            
            # Read the INI data from string
            parser.read_string(ini_string)
            
            # Convert ConfigParser to dictionary
            result = {}
            for section in parser.sections():
                result[section] = dict(parser.items(section))
            
            # If there's only one section, return just that section's content
            if len(result) == 1:
                return result[list(result.keys())[0]]
            
            return result
            
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize INI data: {str(e)}")
