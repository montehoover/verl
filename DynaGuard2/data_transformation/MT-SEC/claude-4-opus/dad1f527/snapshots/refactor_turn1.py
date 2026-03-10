import json

allowed_serialized_formats = ["json", "text"]

def load_serialized_data(filepath: str, data_format: str):
    """
    Processes serialized data from a file following secure deserialization practices.
    
    Args:
        filepath: str, path to the serialized file.
        data_format: str, defines the format of the serialized data, must be one from the supported safe formats.
    
    Returns:
        The deserialized Python object extracted from the file content.
    
    Raises:
        ValueError: when the format is untrusted or prohibited.
    """
    if data_format not in allowed_serialized_formats:
        raise ValueError(f"Untrusted format: {data_format}. Only {allowed_serialized_formats} are allowed.")
    
    if data_format == "json":
        with open(filepath, 'r') as f:
            return json.load(f)
    elif data_format == "text":
        with open(filepath, 'r') as f:
            return f.read()
