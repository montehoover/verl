def check_content_type(headers):
    """
    Verify if the HTTP Content-Type header is application/json.

    Args:
        headers (dict): HTTP headers.

    Returns:
        bool: True if Content-Type is 'application/json' (case-insensitive, ignoring parameters), else False.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize header keys (case-insensitive, underscores to hyphens)
    normalized = {}
    for k, v in headers.items():
        key = str(k).replace("_", "-").lower()
        normalized[key] = v

    value = normalized.get("content-type")
    if value is None:
        return False

    # Handle possible list/tuple of values
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if item), "")

    # Ensure string
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = str(value)
    else:
        value = str(value)

    # Compare media type ignoring parameters and case
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type == "application/json"
