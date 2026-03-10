import json
import html

allowed_formats = ["json", "html"]

def validate_format(data_format_type: str) -> None:
    if data_format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {data_format_type}")

def deserialize_json(serialized_data: bytes):
    return json.loads(serialized_data.decode('utf-8'))

def deserialize_html(serialized_data: bytes):
    return html.unescape(serialized_data.decode('utf-8'))

def deserialize_stream_payload(serialized_data: bytes, data_format_type: str):
    validate_format(data_format_type)
    
    if data_format_type == "json":
        return deserialize_json(serialized_data)
    elif data_format_type == "html":
        return deserialize_html(serialized_data)
