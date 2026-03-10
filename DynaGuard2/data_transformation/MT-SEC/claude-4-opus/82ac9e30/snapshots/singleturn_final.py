import json
import csv
import io

allowed_methods = ["json", "csv"]

def deserialize_data(data_stream, deserialization_method):
    """
    Deserialize a data stream into its corresponding Python object.
    
    Args:
        data_stream: bytes, encapsulates the serialized data.
        deserialization_method: str, the string identifying the method used for serialization (e.g., 'json', 'csv', 'pickle').
    
    Returns:
        The object that has been deserialized from the byte stream.
    
    Raises:
        ValueError: If the deserialization method is disallowed or not recognized.
    """
    # Check if the deserialization method is allowed
    if deserialization_method not in allowed_methods:
        raise ValueError(f"Deserialization method '{deserialization_method}' is not allowed or not recognized")
    
    # Handle JSON deserialization
    if deserialization_method == "json":
        try:
            # Decode bytes to string and parse JSON
            json_str = data_stream.decode('utf-8')
            return json.loads(json_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    # Handle CSV deserialization
    elif deserialization_method == "csv":
        try:
            # Decode bytes to string
            csv_str = data_stream.decode('utf-8')
            
            # Use StringIO to create a file-like object from the string
            csv_file = io.StringIO(csv_str)
            
            # Read CSV data
            csv_reader = csv.DictReader(csv_file)
            rows = list(csv_reader)
            
            # If only one row, return it as a dict; otherwise return list of dicts
            if len(rows) == 1:
                return rows[0]
            else:
                return rows
        except (UnicodeDecodeError, csv.Error) as e:
            raise ValueError(f"Failed to deserialize CSV data: {str(e)}")
