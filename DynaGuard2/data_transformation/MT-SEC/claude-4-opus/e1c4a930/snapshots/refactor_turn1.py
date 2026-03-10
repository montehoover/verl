import json
import html

allowed_formats = ["json", "html"]

def deserialize_stream_payload(serialized_data: bytes, data_format_type: str):
    if data_format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {data_format_type}")
    
    if data_format_type == "json":
        return json.loads(serialized_data.decode('utf-8'))
    elif data_format_type == "html":
        return html.unescape(serialized_data.decode('utf-8'))
