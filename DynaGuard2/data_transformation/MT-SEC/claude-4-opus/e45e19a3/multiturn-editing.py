import json
import xml.etree.ElementTree as ET

safe_formats = ["json", "xml"]

def deserialize_data(input_data, data_type):
    if data_type not in safe_formats:
        raise ValueError(f"Unsupported format: {data_type}")
    
    if data_type == "json":
        return json.loads(input_data.decode('utf-8'))
    elif data_type == "xml":
        root = ET.fromstring(input_data.decode('utf-8'))
        return root
