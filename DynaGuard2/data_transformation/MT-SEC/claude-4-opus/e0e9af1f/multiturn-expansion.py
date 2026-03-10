import json

def read_file_as_bytes(file_path):
    """
    Read a file and return its contents as bytes.
    
    Args:
        file_path (str): The path to the file to read.
        
    Returns:
        bytes: The contents of the file as bytes.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    with open(file_path, 'rb') as file:
        return file.read()


def validate_file_format(file_path, format_string):
    """
    Validate if a file's content matches the specified format.
    
    Args:
        file_path (str): The path to the file to validate.
        format_string (str): The expected format ('json' or 'text').
        
    Returns:
        bool: True if the file matches the specified format, False otherwise.
        
    Raises:
        ValueError: If the format_string is unrecognized or potentially unsafe.
    """
    allowed_formats = ['json', 'text']
    
    if format_string not in allowed_formats:
        raise ValueError(f"Unrecognized or potentially unsafe format: {format_string}")
    
    try:
        file_content = read_file_as_bytes(file_path)
        
        if format_string == 'json':
            try:
                # Try to decode and parse as JSON
                json.loads(file_content.decode('utf-8'))
                return True
            except (json.JSONDecodeError, UnicodeDecodeError):
                return False
                
        elif format_string == 'text':
            try:
                # Try to decode as UTF-8 text
                file_content.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
                
    except (FileNotFoundError, IOError):
        return False


def load_serialized_data(file_location, data_format):
    """
    Securely deserialize data from a file.
    
    Args:
        file_location (str): The path to the file containing serialized data.
        data_format (str): The format of the serialized data ('json' or 'text').
        
    Returns:
        The deserialized Python object.
        
    Raises:
        ValueError: If the format is unsafe or unsupported.
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    acceptable_formats = ["json", "text"]
    
    if data_format not in acceptable_formats:
        raise ValueError(f"Unsafe or unsupported format: {data_format}")
    
    if not validate_file_format(file_location, data_format):
        raise ValueError(f"File content does not match the specified format: {data_format}")
    
    file_content = read_file_as_bytes(file_location)
    
    if data_format == 'json':
        return json.loads(file_content.decode('utf-8'))
    elif data_format == 'text':
        return file_content.decode('utf-8')
