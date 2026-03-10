import json
import html

accepted_formats = ["json", "html"]

def validate_format(data_format: str) -> None:
    if data_format not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")

def deserialize_json(data: bytes):
    return json.loads(data.decode('utf-8'))

def deserialize_html(data: bytes):
    return html.unescape(data.decode('utf-8'))

def get_deserializer(data_format: str):
    deserializers = {
        "json": deserialize_json,
        "html": deserialize_html
    }
    return deserializers[data_format]

def decode_data_stream(data: bytes, data_format: str):
    validate_format(data_format)
    deserializer = get_deserializer(data_format)
    return deserializer(data)
