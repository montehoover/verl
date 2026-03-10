import json
import xml.etree.ElementTree as ET

trusted_formats = ["json", "text"]

def read_file_contents(file_path):
    """
    Read and return the contents of a file as a string.
    
    Args:
        file_path (str): Path to the file to read
        
    Returns:
        str: Contents of the file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def detect_file_format(file_path):
    """
    Detect the format of a file based on its content.
    
    Args:
        file_path (str): Path to the file to analyze
        
    Returns:
        str: File format ('json', 'xml', or 'text')
        
    Raises:
        ValueError: If format is unrecognized or potentially harmful
    """
    try:
        content = read_file_contents(file_path)
        
        # Check for empty file
        if not content.strip():
            raise ValueError("Empty file")
        
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
        
        # Check for potentially harmful content
        harmful_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
        for pattern in harmful_patterns:
            if pattern.lower() in content.lower():
                raise ValueError(f"Potentially harmful content detected: {pattern}")
        
        # Default to text if no other format matches
        return 'text'
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error reading or detecting file format: {str(e)}")

def read_serialized_file(file_path, file_format):
    """
    Safely deserialize the contents of a file.
    
    Args:
        file_path (str): Path to the file to read
        file_format (str): Serialization format ('json' or 'text')
        
    Returns:
        Deserialized Python object
        
    Raises:
        ValueError: If serialization format is unsafe or unsupported
    """
    if file_format not in trusted_formats:
        raise ValueError(f"Unsupported or unsafe format: {file_format}")
    
    content = read_file_contents(file_path)
    
    if file_format == 'json':
        return json.loads(content)
    elif file_format == 'text':
        return content
    else:
        raise ValueError(f"Unsupported format: {file_format}")
