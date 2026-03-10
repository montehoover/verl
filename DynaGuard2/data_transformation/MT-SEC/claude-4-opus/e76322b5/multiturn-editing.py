import json
import csv
from io import StringIO

supported_formats = ["json", "csv"]

def decode_data(encoded_bytes, format_name):
    if format_name not in supported_formats:
        raise ValueError(f"Unsupported format: {format_name}. Supported formats are: {', '.join(supported_formats)}")
    
    decoded_string = encoded_bytes.decode('utf-8')
    
    if format_name == 'json':
        return json.loads(decoded_string)
    elif format_name == 'csv':
        reader = csv.DictReader(StringIO(decoded_string))
        return list(reader)
