import json
import csv
import io

trusted_serializations = ["json", "csv"]

def restore_object(serialized_bytes, serialization_method):
    """
    Restores a Python object from a serialized data stream.
    
    Args:
        serialized_bytes: bytes, the data stream to deserialize.
        serialization_method: str, identifies the format used for serialization (e.g. 'json', 'csv', 'pickle').
    
    Returns:
        A Python object restored from the serialized data.
    
    Raises:
        ValueError: if the method is unrecognized or insecure.
    """
    # Check if the serialization method is trusted
    if serialization_method not in trusted_serializations:
        raise ValueError(f"Untrusted or unrecognized serialization method: {serialization_method}")
    
    # Deserialize based on the method
    if serialization_method == "json":
        try:
            # Decode bytes to string and parse JSON
            json_string = serialized_bytes.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON: {str(e)}")
    
    elif serialization_method == "csv":
        try:
            # Decode bytes to string and parse CSV
            csv_string = serialized_bytes.decode('utf-8')
            # Use StringIO to create a file-like object from the string
            csv_file = io.StringIO(csv_string)
            reader = csv.DictReader(csv_file)
            # Return list of dictionaries for CSV data
            return list(reader)
        except (UnicodeDecodeError, csv.Error) as e:
            raise ValueError(f"Failed to deserialize CSV: {str(e)}")
    
    # This should not be reached due to the initial check, but included for completeness
    else:
        raise ValueError(f"Unsupported serialization method: {serialization_method}")
