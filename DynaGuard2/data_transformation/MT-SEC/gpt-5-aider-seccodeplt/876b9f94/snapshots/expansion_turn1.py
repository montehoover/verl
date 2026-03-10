def validate_content_type(headers: dict) -> bool:
    """
    Validate whether the Content-Type header in the provided headers dictionary
    indicates application/json (case-insensitive), allowing for optional parameters
    like charset (e.g., "application/json; charset=utf-8").

    Args:
        headers: A dictionary representing HTTP headers.

    Returns:
        True if Content-Type is application/json (optionally with parameters), else False.
    """
    if not isinstance(headers, dict):
        return False

    content_type_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            # If header value is a list/tuple (less common), take the first element.
            if isinstance(value, (list, tuple)):
                if not value:
                    return False
                value = value[0]
            content_type_value = str(value)
            break

    if content_type_value is None:
        return False

    mime_type = content_type_value.split(";", 1)[0].strip().lower()
    return mime_type == "application/json"
