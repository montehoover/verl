def validate_content_type(headers: dict) -> bool:
    """
    Validate that the incoming request headers specify Content-Type: application/json.

    - Header name matching is case-insensitive.
    - Parameters like `; charset=utf-8` are allowed.
    """
    if not isinstance(headers, dict):
        return False

    content_type_value = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == "content-type":
            content_type_value = v
            break

    if content_type_value is None:
        return False

    # If value is list-like, use the first element.
    if isinstance(content_type_value, (list, tuple)):
        content_type_value = content_type_value[0] if content_type_value else ""

    if not isinstance(content_type_value, str):
        # Try to coerce to string
        try:
            content_type_value = str(content_type_value)
        except Exception:
            return False

    media_type = content_type_value.split(";")[0].strip().lower()

    return media_type == "application/json"
