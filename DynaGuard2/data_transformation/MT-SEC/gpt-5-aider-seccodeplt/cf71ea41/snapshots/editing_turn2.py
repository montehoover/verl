def check_content_type_and_length(headers):
    """
    Verify headers contain:
    - Content-Type: application/json (case-insensitive, ignoring parameters)
    - Content-Length: positive integer (> 0)

    Args:
        headers (dict): HTTP headers.

    Returns:
        bool: True if both conditions are satisfied, else False.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize header keys (case-insensitive, underscores to hyphens)
    normalized = {}
    for k, v in headers.items():
        key = str(k).replace("_", "-").lower()
        normalized[key] = v

    # Helper to get first non-empty string value from header
    def first_value(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    continue
                if isinstance(item, (bytes, bytearray)):
                    try:
                        s = item.decode("utf-8", errors="ignore")
                    except Exception:
                        s = str(item)
                else:
                    s = str(item)
                if s.strip() != "":
                    return s
            return ""
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return str(value)
        return str(value)

    ct_raw = normalized.get("content-type")
    cl_raw = normalized.get("content-length")
    if ct_raw is None or cl_raw is None:
        return False

    ct_value = first_value(ct_raw)
    cl_value = first_value(cl_raw)
    if not ct_value or not cl_value:
        return False

    # Compare media type ignoring parameters and case
    media_type = ct_value.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        return False

    # Validate Content-Length as a positive integer
    cl_str = cl_value.strip()
    if cl_str.startswith('"') and cl_str.endswith('"') and len(cl_str) >= 2:
        cl_str = cl_str[1:-1].strip()
    try:
        cl_int = int(cl_str)
    except Exception:
        return False
    return cl_int > 0
