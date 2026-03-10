import json
import xml.etree.ElementTree as ET

allowed_serialized_formats = ["json", "text"]

def read_file_content(file_path):
    """Read the content of a file and return it as a string.
    
    Args:
        file_path (str): The path to the file to read.
        
    Returns:
        str: The content of the file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def validate_file_format(content):
    """Validate and determine the format of file content.
    
    Args:
        content (str): The file content as a string.
        
    Returns:
        str: The detected format ('json', 'xml', or 'text').
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe.
    """
    # Check if content is empty
    if not content or not content.strip():
        raise ValueError("File content is empty")
    
    # Try to parse as JSON
    try:
        json.loads(content)
        return 'json'
    except json.JSONDecodeError:
        pass
    
    # Try to parse as XML
    try:
        ET.fromstring(content)
        return 'xml'
    except ET.ParseError:
        pass
    
    # Check for plain text characteristics
    # If it's not JSON or XML, check if it's reasonable plain text
    if content.isprintable() or all(char.isprintable() or char.isspace() for char in content):
        return 'text'
    
    # If none of the above, raise an error
    raise ValueError("Unrecognized or potentially unsafe file format")

def load_serialized_data(filepath, data_format):
    """Securely deserialize data from a file.
    
    Args:
        filepath (str): The path to the file to load.
        data_format (str): The format of the serialized data ('json' or 'text').
        
    Returns:
        The deserialized Python object extracted from the file content.
        
    Raises:
        ValueError: When the format is untrusted or prohibited.
    """
    # Check if the format is allowed
    if data_format not in allowed_serialized_formats:
        raise ValueError(f"Untrusted or prohibited format: {data_format}")
    
    # Read the file content
    content = read_file_content(filepath)
    
    # Deserialize based on format
    if data_format == 'json':
        return json.loads(content)
    elif data_format == 'text':
        return content
    else:
        raise ValueError(f"Unsupported format: {data_format}")
