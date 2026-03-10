import json
import csv
import io
import logging

# Configure logging
logger = logging.getLogger(__name__)

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
        logger.error(f"Validation failed: Unsupported format type '{format_type}'")
        raise ValueError(f"Unsupported or insecure serialization format: {format_type}")
    logger.debug(f"Format validation successful for '{format_type}'")

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
        result = json.loads(raw_data.decode('utf-8'))
        logger.debug(f"JSON deserialization successful, resulted in {type(result).__name__} object")
        return result
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"JSON deserialization failed: {type(e).__name__}: {e}")
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
        result = list(csv_reader)
        logger.debug(f"CSV deserialization successful, resulted in list with {len(result)} rows")
        return result
    except (UnicodeDecodeError, csv.Error) as e:
        logger.error(f"CSV deserialization failed: {type(e).__name__}: {e}")
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
    logger.info(f"Deserialization attempt started for format '{format_type}' with {len(raw_data)} bytes of data")
    
    try:
        validate_format(format_type)
        
        if format_type == "json":
            result = deserialize_json(raw_data)
        elif format_type == "csv":
            result = deserialize_csv(raw_data)
        
        logger.info(f"Deserialization successful for format '{format_type}'")
        return result
    except ValueError as e:
        logger.error(f"Deserialization failed for format '{format_type}': {e}")
        raise
    except Exception as e:
        logger.critical(f"Unexpected error during deserialization for format '{format_type}': {type(e).__name__}: {e}")
        raise
