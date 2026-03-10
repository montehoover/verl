import json
import csv
import io
from typing import Union, List, Dict, Any

def decode_data(encoded_bytes: bytes, format_type: str) -> Union[str, Dict[str, Any], List[Dict[str, str]]]:
    """
    Decodes a byte stream to a string using UTF-8 encoding and
    then parses it based on the specified format_type.

    Args:
        encoded_bytes: The bytes object to decode.
        format_type: A string indicating the format of the decoded data,
                     either 'json' or 'csv'.

    Returns:
        A Python dictionary if format_type is 'json' and parsing is successful.
        A list of dictionaries if format_type is 'csv' and parsing is successful.
        The decoded string if format_type is not 'json' or 'csv',
        or if JSON parsing fails.
    """
    decoded_string = encoded_bytes.decode('utf-8')

    if format_type == 'json':
        try:
            return json.loads(decoded_string)
        except json.JSONDecodeError:
            # If JSON decoding fails, return the original string as per "whenever possible"
            return decoded_string
    elif format_type == 'csv':
        # Use io.StringIO to treat the string as a file for csv.DictReader
        csv_file = io.StringIO(decoded_string)
        reader = csv.DictReader(csv_file)
        return list(reader)
    else:
        return decoded_string
