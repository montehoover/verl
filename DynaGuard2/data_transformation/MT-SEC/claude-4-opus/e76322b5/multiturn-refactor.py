import json
import csv
import io
import logging

# Configure logging
logger = logging.getLogger(__name__)

supported_formats = ["json", "csv"]

def decode_json(encoded_bytes):
    """
    Deserialize JSON data from bytes.
    
    Args:
        encoded_bytes: bytes, representing the JSON serialized input.
    
    Returns:
        The deserialized Python object.
    
    Raises:
        ValueError if decoding fails.
    """
    try:
        decoded_string = encoded_bytes.decode('utf-8')
        return json.loads(decoded_string)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to decode JSON: {e}")

def decode_csv(encoded_bytes):
    """
    Deserialize CSV data from bytes.
    
    Args:
        encoded_bytes: bytes, representing the CSV serialized input.
    
    Returns:
        List of dictionaries representing the CSV rows.
    
    Raises:
        ValueError if decoding fails.
    """
    try:
        decoded_string = encoded_bytes.decode('utf-8')
        csv_file = io.StringIO(decoded_string)
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        return rows
    except (UnicodeDecodeError, csv.Error) as e:
        raise ValueError(f"Failed to decode CSV: {e}")

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
    logger.info(f"Attempting to decode data with format: {format_name}")
    
    if format_name not in supported_formats:
        logger.error(f"Unsupported or insecure serialization format: {format_name}")
        raise ValueError(f"Unsupported or insecure serialization format: {format_name}")
    
    format_decoders = {
        "json": decode_json,
        "csv": decode_csv
    }
    
    decoder = format_decoders[format_name]
    
    try:
        result = decoder(encoded_bytes)
        logger.info(f"Successfully decoded {format_name} data")
        return result
    except ValueError as e:
        logger.error(f"Failed to decode {format_name} data: {e}")
        raise
