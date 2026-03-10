import json
import html

allowed_formats = ["json", "html"]

def deserialize_stream(serialized_data: bytes, stream_format: str):
    if stream_format not in allowed_formats:
        raise ValueError(f"Format '{stream_format}' is either unsupported or considered insecure")
    
    if stream_format == "json":
        try:
            return json.loads(serialized_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    elif stream_format == "html":
        try:
            decoded_str = serialized_data.decode('utf-8')
            return html.unescape(decoded_str)
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to deserialize HTML data: {str(e)}")
