import re

def include_custom_header(custom_value: str) -> str:
    """
    Insert a custom header into the response headers by returning the full header line.

    The provided value is sanitized to be suitable for an HTTP header field value:
      - CR and LF characters are removed to prevent header injection.
      - Tabs are normalized to a single space.
      - All remaining control characters (0x00-0x1F and 0x7F) are removed.
      - Multiple spaces are collapsed into a single space and the result is trimmed.

    Args:
        custom_value (str): The user-defined value for the custom header.

    Returns:
        str: The full header string, e.g., "Custom-Header: <sanitized_value>".

    Notes:
        This function does not explicitly raise custom exceptions. However, regex-related
        issues could potentially raise 're.error'.
    """
    # Ensure we are working with a string
    value = str(custom_value)

    # Remove CR/LF to prevent header injection
    value = value.replace("\r", "").replace("\n", "")

    # Normalize tabs to a single space
    value = re.sub(r"\t+", " ", value)

    # Remove remaining control characters (including DEL 0x7F), keep printable and extended bytes
    value = re.sub(r"[\x00-\x1F\x7F]", "", value)

    # Collapse multiple spaces and trim
    value = re.sub(r" {2,}", " ", value).strip()

    return f"Custom-Header: {value}"
