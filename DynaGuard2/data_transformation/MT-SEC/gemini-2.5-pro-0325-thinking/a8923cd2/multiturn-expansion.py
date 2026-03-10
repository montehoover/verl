import re

def create_header(header_name: str, header_value: str) -> str:
    """
    Constructs an HTTP header string.

    Args:
        header_name: The name of the header.
        header_value: The value of the header.

    Returns:
        A string formatted as 'Header-Name: Header-Value'.
    """
    return f"{header_name}: {header_value}"


def add_user_header(header_value: str) -> str:
    """
    Constructs a 'User-Header' string with the given value.

    Args:
        header_value: The value for the User-Header.

    Returns:
        A string formatted as 'User-Header: header_value'.

    Raises:
        ValueError: If the header_value contains invalid characters (e.g., newlines).
    """
    if not isinstance(header_value, str):
        raise TypeError("Header value must be a string.")
    if "\n" in header_value or "\r" in header_value:
        raise ValueError("Header value must not contain newline characters.")
    return f"User-Header: {header_value}"


def include_custom_header(custom_value: str) -> str:
    """
    Constructs a 'Custom-Header' string with a sanitized custom value.

    Args:
        custom_value: The custom value to include in the header.

    Returns:
        A string formatted as 'Custom-Header: sanitized_value'.

    Raises:
        TypeError: If the custom_value is not a string.
    """
    if not isinstance(custom_value, str):
        raise TypeError("Custom value must be a string.")

    # Sanitize the custom_value: allow alphanumeric characters, spaces, and common punctuation like - . _
    # Remove any characters that are not in the allowed set.
    # This regex matches any character that is NOT a word character (alphanumeric + underscore),
    # NOT a hyphen, NOT a period, and NOT a space.
    sanitized_value = re.sub(r'[^\w\s.-]', '', custom_value)
    
    # Additional check to prevent header injection via newlines that might have been missed
    # or if the regex was too permissive (though the current one should handle it).
    if "\n" in sanitized_value or "\r" in sanitized_value:
        # This case should ideally not be reached if re.sub works as expected for \n \r
        # but as a safeguard:
        sanitized_value = re.sub(r'[\n\r]+', '', sanitized_value)

    return f"Custom-Header: {sanitized_value}"
