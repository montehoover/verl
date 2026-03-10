import json
import csv
import io

supported_formats = ["json", "csv"]

def decode_data(encoded_bytes, format_name):
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
    if format_name not in supported_formats:
        raise ValueError(f"Unsupported or insecure serialization format: {format_name}")
    
    if format_name == "json":
        try:
            decoded_string = encoded_bytes.decode('utf-8')
            return json.loads(decoded_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to decode JSON: {e}")
    
    elif format_name == "csv":
        try:
            decoded_string = encoded_bytes.decode('utf-8')
            csv_file = io.StringIO(decoded_string)
            reader = csv.DictReader(csv_file)
            rows = list(reader)
            return rows
        except (UnicodeDecodeError, csv.Error) as e:
            raise ValueError(f"Failed to decode CSV: {e}")
