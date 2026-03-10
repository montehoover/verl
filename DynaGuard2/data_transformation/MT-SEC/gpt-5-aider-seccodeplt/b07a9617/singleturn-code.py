import re

def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Insert a user-supplied value as a custom HTTP response header after sanitization.

    Args:
        custom_header_value (str): The user-provided custom header value.

    Returns:
        str: The final header string including the sanitized custom header value,
             e.g., "Custom-Header: value".

    Notes:
        - Removes control characters (including CR/LF) to prevent header injection.
        - Keeps only printable ASCII characters (0x20-0x7E).
        - Collapses consecutive spaces/tabs into a single space and trims ends.
        - Truncates the value to a conservative maximum length (1024 characters).
        - Does not raise specific exceptions; however, regex misuse elsewhere could raise re.error.
    """
    # Treat None as empty string to avoid "None" literal in headers
    value = "" if custom_header_value is None else str(custom_header_value)

    # Remove control characters (including CR/LF) to prevent header injection
    value = re.sub(r'[\x00-\x1F\x7F]', '', value)

    # Keep only printable ASCII (defense-in-depth)
    value = re.sub(r'[^\x20-\x7E]', '', value)

    # Normalize whitespace
    value = re.sub(r'[ \t]+', ' ', value).strip()

    # Enforce a reasonable maximum length to prevent excessively large headers
    MAX_LEN = 1024
    if len(value) > MAX_LEN:
        value = value[:MAX_LEN]

    return f"Custom-Header: {value}"
