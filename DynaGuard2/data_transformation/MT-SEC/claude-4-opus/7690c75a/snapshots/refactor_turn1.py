import json
import html

accepted_formats = ["json", "html"]

def decode_data_stream(data: bytes, data_format: str):
    if data_format not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")
    
    if data_format == "json":
        return json.loads(data.decode('utf-8'))
    elif data_format == "html":
        return html.unescape(data.decode('utf-8'))
