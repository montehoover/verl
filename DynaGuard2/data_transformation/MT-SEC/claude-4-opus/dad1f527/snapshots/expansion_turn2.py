import json
import xml.etree.ElementTree as ET

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
