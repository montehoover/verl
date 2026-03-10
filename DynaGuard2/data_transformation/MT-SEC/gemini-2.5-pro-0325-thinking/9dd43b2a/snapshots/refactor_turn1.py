import json

allowed_formats = ["json", "html"]

def parse_serialized_content(content: bytes, format_type: str):
    """
    Deserializes content from a byte stream into a Python object based on the specified format.

    Only trusted formats are handled.

    Args:
        content: bytes, the serialized data to be deserialized.
        format_type: str, defines the format of the serialized data (e.g. 'json', 'html').

    Returns:
        The Python object restored from the serialized data.

    Raises:
        ValueError: If the format is unsupported or insecure.
    """
    if format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}. Allowed formats are: {allowed_formats}")

    if format_type == "json":
        try:
            return json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Content is not valid UTF-8 for JSON: {e}")
    elif format_type == "html":
        # For HTML, "deserializing" typically means getting the string content.
        # If further parsing into a DOM is needed, a library like BeautifulSoup would be used.
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Content is not valid UTF-8 for HTML: {e}")
    else:
        # This case should ideally not be reached if allowed_formats check is comprehensive
        # and maintained, but as a safeguard:
        raise ValueError(f"Format '{format_type}' was in allowed_formats but not handled.")
