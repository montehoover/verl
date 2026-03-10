import json
import html

safe_formats = ["json", "html"]

def deserialize_json(byte_data: bytes) -> object:
    return json.loads(byte_data.decode('utf-8'))

def deserialize_html(byte_data: bytes) -> str:
    return html.unescape(byte_data.decode('utf-8'))

def restore_object_from_stream(byte_data: bytes, serialization_type: str):
    if serialization_type not in safe_formats:
        raise ValueError(f"Unsupported or unsafe format: {serialization_type}")
    
    deserializers = {
        "json": deserialize_json,
        "html": deserialize_html
    }
    
    return deserializers[serialization_type](byte_data)
