import json
import csv
from io import StringIO

def decode_data(encoded_bytes, format_type=None):
    decoded_string = encoded_bytes.decode('utf-8')
    
    if format_type == 'json':
        return json.loads(decoded_string)
    elif format_type == 'csv':
        reader = csv.DictReader(StringIO(decoded_string))
        return list(reader)
    else:
        try:
            return json.loads(decoded_string)
        except json.JSONDecodeError:
            return decoded_string
