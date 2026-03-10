import re


def format_header(name, value):
    return f"{name}: {value}"


def validate_header_value(header_value):
    # Check for invalid characters (newline, carriage return, null)
    if '\n' in header_value or '\r' in header_value or '\0' in header_value:
        return False
    return True


def append_user_header(user_header):
    # Check for invalid characters using regex
    if re.search(r'[\r\n\0]', user_header):
        # Return empty string or some safe default if invalid
        return "Custom-Header: "
    return f"Custom-Header: {user_header}"
