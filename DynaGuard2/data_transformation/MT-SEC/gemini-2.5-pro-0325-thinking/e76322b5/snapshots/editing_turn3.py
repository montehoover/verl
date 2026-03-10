import json
import csv
import io
from typing import Union, List, Dict, Any

supported_formats = ["json", "csv"]

def decode_data(encoded_bytes: bytes, format_name: str) -> Union[Dict[str, Any], List[Dict[str, str]]]:
    """
    Decodes a byte stream using UTF-8 and deserializes it based on the specified format_name.

    Only permits deserialization methods listed in `supported_formats`.

    Args:
        encoded_bytes: The bytes object to decode and deserialize.
        format_name: The name of the format to use for deserialization (e.g., "json", "csv").

    Returns:
        The deserialized Python object (e.g., a dictionary for JSON, a list of dictionaries for CSV).

    Raises:
        ValueError: If `format_name` is not in `supported_formats`.
        json.JSONDecodeError: If `format_name` is "json" and the data is not valid JSON.
        # Other parsing errors for CSV might also propagate.
    """
    if format_name not in supported_formats:
        raise ValueError(f"Unsupported format: '{format_name}'. Supported formats are: {supported_formats}")

    decoded_string = encoded_bytes.decode('utf-8')

    if format_name == 'json':
        return json.loads(decoded_string)
    elif format_name == 'csv':
        # Use io.StringIO to treat the string as a file for csv.DictReader
        csv_file = io.StringIO(decoded_string)
        reader = csv.DictReader(csv_file)
        return list(reader)
    # All supported formats are handled by the if/elif chain.
    # An unsupported format would have raised a ValueError earlier.
