import re

def sanitize_input(user_input: str) -> str:
    """
    Sanitizes user input by removing characters not allowed in HTTP headers.
    Allows printable ASCII characters (space through tilde).

    Args:
        user_input: The string to sanitize.

    Returns:
        The sanitized string.
    """
    # Remove any character that is not a printable ASCII character (space to ~)
    return re.sub(r'[^ -~]', '', user_input)

def create_http_header(header_name: str) -> str:
    """
    Creates a basic HTTP header string.

    Args:
        header_name: The name of the header (e.g., 'Content-Type').

    Returns:
        A string formatted as an HTTP header (e.g., 'Content-Type: default_value').
    """
    # Format the header name: capitalize words and join with hyphens
    formatted_header_name = '-'.join(word.capitalize() for word in header_name.replace('-', ' ').split())
    return f"{formatted_header_name}: default_value"

def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Creates a custom HTTP header string with a sanitized value.

    Args:
        custom_header_value: The value for the custom header.

    Returns:
        A string formatted as 'X-Custom-User-Header: <sanitized_value>'.
    """
    sanitized_value = sanitize_input(custom_header_value)
    header_name = "X-Custom-User-Header"
    return f"{header_name}: {sanitized_value}"

if __name__ == '__main__':
    # Example usage:
    header1 = create_http_header("Content-Type")
    print(f"'{header1}'")

    header2 = create_http_header("user agent")
    print(f"'{header2}'")

    header3 = create_http_header("X-Custom-Header")
    print(f"'{header3}'")

    # Example usage for sanitize_input:
    dirty_input1 = "Valid-Input123"
    clean_input1 = sanitize_input(dirty_input1)
    print(f"Sanitizing '{dirty_input1}': '{clean_input1}'")

    dirty_input2 = "Input with\r\nnewlines and\ttabs"
    clean_input2 = sanitize_input(dirty_input2)
    print(f"Sanitizing '{dirty_input2}': '{clean_input2}'")

    dirty_input3 = "Input with non-ASCII: éàçüö"
    clean_input3 = sanitize_input(dirty_input3)
    print(f"Sanitizing '{dirty_input3}': '{clean_input3}'")

    dirty_input4 = "Another\0NullCharInput"
    clean_input4 = sanitize_input(dirty_input4)
    print(f"Sanitizing '{dirty_input4}': '{clean_input4}'")

    # Example usage for insert_header_with_sanitization:
    custom_val1 = "My custom value"
    custom_header1 = insert_header_with_sanitization(custom_val1)
    print(f"Custom header with '{custom_val1}': '{custom_header1}'")

    custom_val2 = "Value with\r\nCRLF and\tTabs"
    custom_header2 = insert_header_with_sanitization(custom_val2)
    print(f"Custom header with '{custom_val2}': '{custom_header2}'")

    custom_val3 = "Another value with non-ASCII: éàçüö and control\0chars"
    custom_header3 = insert_header_with_sanitization(custom_val3)
    print(f"Custom header with '{custom_val3}': '{custom_header3}'")
