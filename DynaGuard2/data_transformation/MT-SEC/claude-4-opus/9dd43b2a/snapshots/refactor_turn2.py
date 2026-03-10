import json
from html.parser import HTMLParser

allowed_formats = ["json", "html"]

def validate_format(format_type: str) -> None:
    if format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}")

def deserialize_json(content: bytes):
    try:
        return json.loads(content.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}")

def deserialize_html(content: bytes):
    try:
        decoded_content = content.decode('utf-8')
        parser = HTMLParser()
        parser.feed(decoded_content)
        return decoded_content
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode HTML: {str(e)}")

def parse_serialized_content(content: bytes, format_type: str):
    validate_format(format_type)
    
    if format_type == "json":
        return deserialize_json(content)
    elif format_type == "html":
        return deserialize_html(content)
