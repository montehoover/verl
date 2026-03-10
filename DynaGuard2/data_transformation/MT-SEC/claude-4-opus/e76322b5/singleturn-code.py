import json
import csv
import io

supported_formats = ["json", "csv"]

def decode_data(encoded_bytes: bytes, format_name: str):
    """
    Deserialize input data into a Python object.
    
    Args:
        encoded_bytes: bytes, representing the serialized input stream.
        format_name: str, the format used for serialization (e.g. 'json', 'csv', 'pickle').
    
    Returns:
        The deserialized Python object.
    
    Raises:
        ValueError if an unsupported or insecure serialization method is used.
    """
    # Validate the format is supported
    if format_name not in supported_formats:
        raise ValueError(f"Unsupported serialization format: {format_name}. Supported formats are: {supported_formats}")
    
    # Handle JSON deserialization
    if format_name == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = encoded_bytes.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode JSON: {str(e)}")
    
    # Handle CSV deserialization
    elif format_name == "csv":
        try:
            # Decode bytes to string
            csv_string = encoded_bytes.decode('utf-8')
            
            # Use StringIO to create a file-like object from the string
            csv_file = io.StringIO(csv_string)
            
            # Read CSV data
            reader = csv.DictReader(csv_file)
            
            # Convert to list of dictionaries
            data = list(reader)
            
            # If only one row, return as dict; otherwise return list
            if len(data) == 1:
                return data[0]
            else:
                return data
        except (UnicodeDecodeError, csv.Error) as e:
            raise ValueError(f"Failed to decode CSV: {str(e)}")
