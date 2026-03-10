import json
import xml.etree.ElementTree as ET

def convert_json_to_dict(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")

def convert_to_dict(data_string, data_format):
    if data_format == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    elif data_format == 'xml':
        try:
            root = ET.fromstring(data_string)
            return root
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML string: {e}")
    else:
        raise ValueError(f"Unsupported format: {data_format}. Use 'json' or 'xml'.")
