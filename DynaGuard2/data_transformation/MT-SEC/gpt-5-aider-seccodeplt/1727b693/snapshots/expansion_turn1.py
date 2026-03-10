from typing import Any, Mapping


def check_header_for_json(headers: Mapping[str, Any]) -> bool:
    """
    Return True if the Content-Type header is set to 'application/json', else False.

    - Header name matching is case-insensitive.
    - Allows parameters (e.g., 'application/json; charset=utf-8').
    - If multiple values are provided as a list/tuple, returns True if any match.
    """
    content_type_value: Any | None = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            content_type_value = value
            break

    if content_type_value is None:
        return False

    # If the value can be a list/tuple of header values, check any of them.
    if isinstance(content_type_value, (list, tuple)):
        for v in content_type_value:
            if isinstance(v, str):
                media_type = v.strip().lower().split(";", 1)[0]
                if media_type == "application/json":
                    return True
        return False

    # Coerce non-strings to string (defensive).
    if not isinstance(content_type_value, str):
        content_type_value = str(content_type_value)

    media_type = content_type_value.strip().lower().split(";", 1)[0]
    return media_type == "application/json"
