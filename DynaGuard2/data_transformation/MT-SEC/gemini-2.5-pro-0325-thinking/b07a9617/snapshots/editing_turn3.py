import re

def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Sanitizes the custom header value and returns it as part of an HTTP header string.
    Invalid characters (non-printable ASCII) are removed.
    """
    # Sanitize the custom_header_value to allow only printable ASCII characters (0x20-0x7E)
    # This helps prevent header injection attacks by removing CR, LF, and other control characters.
    sanitized_value = re.sub(r'[^\x20-\x7E]', '', custom_header_value)
    return f"X-Custom-Header: {sanitized_value}"
