import json

allowed_formats = ["json", "text"]

def read_file(input_path: str) -> str:
    """Read and return the contents of a file."""
    with open(input_path, 'r') as f:
        return f.read()

def deserialize_json(content: str):
    """Deserialize JSON string content."""
    return json.loads(content)

def deserialize_text(content: str):
    """Return text content as-is."""
    return content

def validate_format(format_type: str) -> None:
    """Validate that the format type is allowed."""
    if format_type not in allowed_formats:
        raise ValueError(f"Untrusted format: {format_type}. Only {allowed_formats} are allowed.")

def process_serialfile(input_path: str, format_type: str):
    """
    Process serialized data from a file following secure deserialization practices.
    
    Args:
        input_path: str, path to the serialized file.
        format_type: str, defines the format of the serialized data, must be one from the supported safe formats.
    
    Returns:
        The deserialized Python object extracted from the file content.
    
    Raises:
        ValueError: when the format is untrusted or prohibited.
    """
    validate_format(format_type)
    
    content = read_file(input_path)
    
    if format_type == "json":
        return deserialize_json(content)
    elif format_type == "text":
        return deserialize_text(content)
