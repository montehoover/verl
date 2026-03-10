import json
import csv
import io

approved_formats = ["json", "csv"]

def deserialize_object(raw_data: bytes, format_type: str):
    """
    Deserialize input data into a Python object.
    
    Args:
        raw_data: bytes, representing the serialized input stream.
        format_type: str, the format used for serialization (e.g. 'json', 'csv', 'pickle').
    
    Returns:
        The deserialized Python object.
    
    Raises:
        ValueError: if an unsupported or insecure serialization method is used.
    """
    # Validate the format type
    if format_type not in approved_formats:
        raise ValueError(f"Unsupported or insecure serialization format: {format_type}")
    
    # Handle JSON deserialization
    if format_type == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = raw_data.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {str(e)}")
    
    # Handle CSV deserialization
    elif format_type == "csv":
        try:
            # Decode bytes to string
            csv_string = raw_data.decode('utf-8')
            
            # Use StringIO to create a file-like object from the string
            csv_file = io.StringIO(csv_string)
            
            # Read CSV data
            reader = csv.DictReader(csv_file)
            rows = list(reader)
            
            # If there's only one row, return it as a dict
            # Otherwise, return the list of rows
            if len(rows) == 1:
                return rows[0]
            else:
                return rows
                
        except (UnicodeDecodeError, csv.Error) as e:
            raise ValueError(f"Invalid CSV data: {str(e)}")
