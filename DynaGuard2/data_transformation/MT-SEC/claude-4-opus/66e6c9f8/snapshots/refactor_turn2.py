import json
from html.parser import HTMLParser

allowed_formats = ["json", "html"]

def validate_format(stream_format):
    if stream_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}")

def deserialize_json(stream):
    try:
        return json.loads(stream.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid JSON data: {e}")

def deserialize_html(stream):
    try:
        decoded_html = stream.decode('utf-8')
        parser = HTMLParser()
        parser.feed(decoded_html)
        return decoded_html
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid HTML data: {e}")

def deserialize_content_stream(stream, stream_format):
    validate_format(stream_format)
    
    if stream_format == "json":
        return deserialize_json(stream)
    elif stream_format == "html":
        return deserialize_html(stream)
