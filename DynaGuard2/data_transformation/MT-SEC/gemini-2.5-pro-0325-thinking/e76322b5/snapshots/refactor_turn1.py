import json
import csv
import io

supported_formats = ["json", "csv"]

def decode_data(encoded_bytes: bytes, format_name: str):
    """
    Deserialize input data into a Python object.

    Args:
        encoded_bytes: bytes, representing the serialized input stream.
        format_name: str, the format used for serialization (e.g. 'json', 'csv').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: if an unsupported or insecure serialization method is used.
    """
    if format_name not in supported_formats:
        raise ValueError(
            f"Unsupported or insecure serialization format: {format_name}. "
            f"Supported formats are: {', '.join(supported_formats)}"
        )

    if format_name == "json":
        try:
            decoded_string = encoded_bytes.decode('utf-8')
            return json.loads(decoded_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding bytes to string for JSON: {e}")
    elif format_name == "csv":
        try:
            decoded_string = encoded_bytes.decode('utf-8')
            # Use io.StringIO to treat the string as a file
            csv_file = io.StringIO(decoded_string)
            reader = csv.reader(csv_file)
            # Convert reader to a list of lists
            return [row for row in reader]
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding bytes to string for CSV: {e}")
        except csv.Error as e:
            raise ValueError(f"Invalid CSV data: {e}")
    
    # This part should ideally not be reached if format_name is validated correctly
    # but as a fallback:
    raise ValueError(f"Deserialization for format '{format_name}' not implemented after validation.")
