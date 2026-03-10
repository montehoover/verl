import json

acceptable_formats = ["json", "text"]

def load_serialized_data(file_location: str, data_format: str):
    """
    Deserialize data from an external file using only secure formats.
    
    Args:
        file_location: str, path to the file containing the serialized data.
        data_format: str, indicates the format of the serialized data, restricted to trusted options.
    
    Returns:
        A Python object that results from deserializing the file contents.
    
    Raises:
        ValueError: if the format is unsafe or unsupported.
    """
    # Check if the format is in the whitelist of acceptable formats
    if data_format not in acceptable_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}. Only {acceptable_formats} are allowed.")
    
    # Handle JSON format
    if data_format == "json":
        with open(file_location, 'r') as f:
            return json.load(f)
    
    # Handle text format
    elif data_format == "text":
        with open(file_location, 'r') as f:
            return f.read()
