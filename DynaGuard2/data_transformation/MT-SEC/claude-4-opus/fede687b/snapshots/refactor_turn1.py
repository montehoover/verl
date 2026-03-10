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
        ValueError if an unsupported or insecure serialization method is used.
    """
    if format_type not in approved_formats:
        raise ValueError(f"Unsupported or insecure serialization format: {format_type}")
    
    if format_type == "json":
        try:
            return json.loads(raw_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    elif format_type == "csv":
        try:
            decoded_data = raw_data.decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(decoded_data))
            return list(csv_reader)
        except (UnicodeDecodeError, csv.Error) as e:
            raise ValueError(f"Invalid CSV data: {e}")
