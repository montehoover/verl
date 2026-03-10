import json
import csv
import io

approved_formats = ["json", "csv"]

def validate_format(format_type: str) -> None:
    """
    Validate that the format type is approved.
    
    Args:
        format_type: str, the format used for serialization.
    
    Raises:
        ValueError if an unsupported or insecure serialization method is used.
    """
    if format_type not in approved_formats:
        raise ValueError(f"Unsupported or insecure serialization format: {format_type}")

def deserialize_json(raw_data: bytes):
    """
    Deserialize JSON data.
    
    Args:
        raw_data: bytes, representing the serialized JSON data.
    
    Returns:
        The deserialized Python object.
    
    Raises:
        ValueError if the JSON data is invalid.
    """
    try:
        return json.loads(raw_data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid JSON data: {e}")

def deserialize_csv(raw_data: bytes):
    """
    Deserialize CSV data.
    
    Args:
        raw_data: bytes, representing the serialized CSV data.
    
    Returns:
        A list of dictionaries representing the CSV rows.
    
    Raises:
        ValueError if the CSV data is invalid.
    """
    try:
        decoded_data = raw_data.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(decoded_data))
        return list(csv_reader)
    except (UnicodeDecodeError, csv.Error) as e:
        raise ValueError(f"Invalid CSV data: {e}")

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
    validate_format(format_type)
    
    if format_type == "json":
        return deserialize_json(raw_data)
    elif format_type == "csv":
        return deserialize_csv(raw_data)
