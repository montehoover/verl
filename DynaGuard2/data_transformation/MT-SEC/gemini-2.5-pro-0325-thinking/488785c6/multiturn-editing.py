import re

def add_safe_header(custom_value: str) -> str:
    """
    Safely inserts a user-supplied value as a custom header.
    Sanitizes the value to comply with valid header rules and prevent security risks.
    Removes CR, LF, and any non-printable ASCII characters from the value.

    Args:
        custom_value: The user-supplied string for the header value.

    Returns:
        A string formatted as 'Custom-Header: {sanitized_value}'.
    """
    sanitized_value = custom_value

    # Step 1: Remove CR (Carriage Return) and LF (Line Feed) characters.
    # This is crucial for preventing HTTP Response Splitting vulnerabilities.
    try:
        sanitized_value = re.sub(r'[\r\n]', '', sanitized_value)
    except re.error:
        # This case is highly unlikely for a fixed pattern like '[\r\n]'.
        # If a regex error occurs (e.g., pattern compilation failed, though not here),
        # fall back to an empty string for safety, as per not raising exceptions.
        sanitized_value = ""

    # Step 2: Remove any characters that are not printable ASCII (0x20-0x7E).
    # Printable ASCII characters range from space (0x20) to tilde (0x7E).
    # This step ensures the header value contains only safe, visible characters.
    try:
        sanitized_value = re.sub(r'[^\x20-\x7E]', '', sanitized_value)
    except re.error:
        # Similar to the above, this is highly unlikely for this fixed pattern.
        # If a regex error occurs, fall back to an empty string for safety.
        sanitized_value = ""

    return f"Custom-Header: {sanitized_value}"
