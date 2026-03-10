from typing import Any, Dict, Iterable


def validate_content_type(headers: Dict[str, Any]) -> bool:
    """
    Validate that the request headers specify Content-Type as application/json.

    - Header name matching is case-insensitive.
    - Accepts parameters (e.g., 'application/json; charset=utf-8').
    - Returns False if the header is missing or does not equal 'application/json'.

    :param headers: A dict representing HTTP request headers.
    :return: True if Content-Type is 'application/json' (optionally with parameters), else False.
    """
    content_type_value: Any = None

    # Find Content-Type header case-insensitively
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "content-type":
            content_type_value = value
            break

    if content_type_value is None:
        return False

    # If header value is a list/tuple, pick the first non-empty string
    if isinstance(content_type_value, (list, tuple)):
        content_type_value = next(
            (v for v in content_type_value if isinstance(v, str) and v.strip()),
            None,
        )
        if content_type_value is None:
            return False

    # Coerce to string if needed
    if not isinstance(content_type_value, str):
        try:
            content_type_value = str(content_type_value)
        except Exception:
            return False

    # Normalize and compare mime type (ignore parameters)
    normalized = content_type_value.strip().lower()
    mime = normalized.split(";", 1)[0].strip()

    return mime == "application/json"
