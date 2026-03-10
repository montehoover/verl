import json
import xml.etree.ElementTree as ET

def convert_string_to_data(data_string, format_type):
    if format_type == 'json':
        return json.loads(data_string)
    elif format_type == 'xml':
        root = ET.fromstring(data_string)
        return root
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
