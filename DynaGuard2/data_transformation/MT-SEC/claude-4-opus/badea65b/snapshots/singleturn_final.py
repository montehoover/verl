import json

allowed_formats = ["json", "text"]

def process_serialfile(input_path: str, format_type: str):
    """
    Process serialized data from a file following secure deserialization practices.
    
    Args:
        input_path: str, path to the serialized file.
        format_type: str, defines the format of the serialized data, must be one of the supported safe formats.
    
    Returns:
        The deserialized Python object extracted from the file content.
    
    Raises:
        ValueError: When the format is untrusted or prohibited.
    """
    # Check if the format is in the allowed formats
    if format_type not in allowed_formats:
        raise ValueError(f"Format '{format_type}' is not allowed. Only {allowed_formats} are permitted.")
    
    # Process based on the format type
    if format_type == "json":
        with open(input_path, 'r') as file:
            return json.load(file)
    
    elif format_type == "text":
        with open(input_path, 'r') as file:
            return file.read()
    
    else:
        # This should not happen due to the check above, but added for safety
        raise ValueError(f"Unsupported format: {format_type}")
