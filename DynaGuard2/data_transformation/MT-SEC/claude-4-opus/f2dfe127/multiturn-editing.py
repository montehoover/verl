import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]

def bytes_to_obj(data_bytes, format_name):
    if format_name not in approved_formats:
        raise ValueError(f"Unsupported format: {format_name}")
    
    if format_name == "json":
        return json.loads(data_bytes.decode('utf-8'))
    elif format_name == "xml":
        return ET.fromstring(data_bytes.decode('utf-8'))
