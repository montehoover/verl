def check_content_type(headers: dict) -> bool:
    """
    Verify if the Content-Type header is application/json (case-insensitive),
    ignoring any parameters like charset.
    """
    if not isinstance(headers, dict):
        return False

    # Find Content-Type header (case-insensitive)
    content_type_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == 'content-type':
            content_type_value = value
            break

    if content_type_value is None:
        return False

    # Support list/tuple header values by taking the first item
    if isinstance(content_type_value, (list, tuple)):
        if not content_type_value:
            return False
        content_type_value = content_type_value[0]

    if not isinstance(content_type_value, str):
        return False

    media_type = content_type_value.split(';', 1)[0].strip().lower()
    return media_type == 'application/json'
