import json
import csv
import io
import logging

trusted_serializations = ["json", "csv"]

# Configure logger
logger = logging.getLogger(__name__)

def deserialize_json(serialized_bytes):
    """Deserialize JSON data from bytes."""
    try:
        result = json.loads(serialized_bytes.decode('utf-8'))
        logger.info("Successfully deserialized JSON data")
        return result
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Failed to deserialize JSON: {e}")
        raise ValueError(f"Failed to deserialize JSON: {e}")

def deserialize_csv(serialized_bytes):
    """Deserialize CSV data from bytes."""
    try:
        csv_string = serialized_bytes.decode('utf-8')
        csv_file = io.StringIO(csv_string)
        reader = csv.DictReader(csv_file)
        result = list(reader)
        logger.info("Successfully deserialized CSV data")
        return result
    except (UnicodeDecodeError, csv.Error) as e:
        logger.error(f"Failed to deserialize CSV: {e}")
        raise ValueError(f"Failed to deserialize CSV: {e}")

def restore_object(serialized_bytes, serialization_method):
    logger.info(f"Attempting to deserialize data using method: {serialization_method}")
    
    if serialization_method not in trusted_serializations:
        logger.warning(f"Rejected untrusted serialization method: {serialization_method}")
        raise ValueError(f"Untrusted or unrecognized serialization method: {serialization_method}")
    
    try:
        if serialization_method == "json":
            result = deserialize_json(serialized_bytes)
        elif serialization_method == "csv":
            result = deserialize_csv(serialized_bytes)
        
        logger.info(f"Deserialization completed successfully using method: {serialization_method}")
        return result
    except ValueError as e:
        logger.error(f"Deserialization failed for method {serialization_method}: {e}")
        raise
