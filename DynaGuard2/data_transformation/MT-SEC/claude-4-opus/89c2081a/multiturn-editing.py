import json
import xml.etree.ElementTree as ET
import configparser
import pickle

secure_serialization_formats = ["json", "ini"]

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

def process_serialized_content(payload_bytes, schema_type):
    if schema_type not in secure_serialization_formats:
        raise ValueError(f"Unsupported or unsafe format: {schema_type}. Supported formats: {secure_serialization_formats}")
    
    if schema_type == 'json':
        try:
            return json.loads(payload_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON: {e}")
    elif schema_type == 'ini':
        try:
            config = configparser.ConfigParser()
            config.read_string(payload_bytes.decode('utf-8'))
            return {section: dict(config.items(section)) for section in config.sections()}
        except (configparser.Error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize INI: {e}")
