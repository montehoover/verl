import json

trusted_formats = ["json", "text"]

def read_serialized_file(file_path: str, file_format: str):
    """
    Safely deserializes the contents of a file, restricted to trusted formats.
    
    Args:
        file_path: str, the location of the serialized data file.
        file_format: str, the serialization format, should be from a trusted list.
    
    Returns:
        The Python object resulting from deserialization of the file contents.
    
    Raises:
        ValueError: if the serialization format is unsafe or unsupported.
    """
    # Check if the format is in the trusted list
    if file_format not in trusted_formats:
        raise ValueError(f"Unsupported or unsafe format: {file_format}. Only {trusted_formats} are allowed.")
    
    # Handle each trusted format
    if file_format == "json":
        with open(file_path, 'r') as f:
            return json.load(f)
    
    elif file_format == "text":
        with open(file_path, 'r') as f:
            return f.read()
    
    else:
        # This should not happen if trusted_formats is properly maintained
        raise ValueError(f"Format {file_format} is in trusted list but not implemented")
