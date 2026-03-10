import json
import html

allowed_formats = ["json", "html"]

def validate_format(stream_format: str) -> None:
    """Validate that the format is allowed and secure."""
    if stream_format not in allowed_formats:
        raise ValueError(f"Format '{stream_format}' is either unsupported or considered insecure")

def deserialize_json(serialized_data: bytes) -> object:
    """Deserialize JSON data from bytes."""
    try:
        return json.loads(serialized_data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to deserialize JSON data: {str(e)}")

def deserialize_html(serialized_data: bytes) -> str:
    """Deserialize HTML data from bytes."""
    try:
        decoded_str = serialized_data.decode('utf-8')
        return html.unescape(decoded_str)
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to deserialize HTML data: {str(e)}")

def deserialize_stream(serialized_data: bytes, stream_format: str):
    validate_format(stream_format)
    
    deserializers = {
        "json": deserialize_json,
        "html": deserialize_html
    }
    
    return deserializers[stream_format](serialized_data)
