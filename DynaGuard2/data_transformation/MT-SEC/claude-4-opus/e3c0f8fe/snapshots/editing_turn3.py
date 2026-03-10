import json
import csv
from io import StringIO

trusted_serializations = ["json", "csv"]

def restore_object(serialized_bytes, serialization_method):
    """Restore a Python object from a serialized byte stream.
    
    Args:
        serialized_bytes (bytes): The serialized data stream
        serialization_method (str): The serialization format ('json', 'csv', 'pickle')
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the method is unrecognized or insecure
    """
    if serialization_method not in trusted_serializations:
        raise ValueError(f"Untrusted or unrecognized serialization method: {serialization_method}")
    
    if serialization_method == 'json':
        try:
            data_string = serialized_bytes.decode('utf-8')
            return json.loads(data_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif serialization_method == 'csv':
        try:
            data_string = serialized_bytes.decode('utf-8')
            csv_file = StringIO(data_string)
            reader = csv.DictReader(csv_file)
            return list(reader)
        except (UnicodeDecodeError, Exception) as e:
            raise ValueError(f"Invalid CSV data: {e}")
