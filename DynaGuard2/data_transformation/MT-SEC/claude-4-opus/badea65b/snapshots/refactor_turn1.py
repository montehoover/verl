import json

allowed_formats = ["json", "text"]

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
    if format_type not in allowed_formats:
        raise ValueError(f"Untrusted format: {format_type}. Only {allowed_formats} are allowed.")
    
    if format_type == "json":
        with open(input_path, 'r') as f:
            return json.load(f)
    elif format_type == "text":
        with open(input_path, 'r') as f:
            return f.read()
