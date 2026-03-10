import json
import html

safe_formats = ["json", "html"]

def restore_object_from_stream(byte_data: bytes, serialization_type: str):
    if serialization_type not in safe_formats:
        raise ValueError(f"Unsupported or unsafe format: {serialization_type}")
    
    if serialization_type == "json":
        return json.loads(byte_data.decode('utf-8'))
    elif serialization_type == "html":
        return html.unescape(byte_data.decode('utf-8'))
