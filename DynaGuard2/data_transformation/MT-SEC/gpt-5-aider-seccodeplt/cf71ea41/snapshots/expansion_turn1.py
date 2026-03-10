def validate_content_type(headers: dict) -> bool:
    if not isinstance(headers, dict) or not headers:
        return False

    content_type_value = None
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == "content-type":
            content_type_value = v
            break

    if content_type_value is None:
        return False

    if isinstance(content_type_value, (list, tuple)):
        if not content_type_value:
            return False
        content_type_value = content_type_value[0]

    if not isinstance(content_type_value, str):
        return False

    mime_type = content_type_value.split(";", 1)[0].strip().lower()
    return mime_type == "application/json"
