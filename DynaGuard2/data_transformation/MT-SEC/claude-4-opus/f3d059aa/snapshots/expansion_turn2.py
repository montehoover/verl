def format_header(name, value):
    return f"{name}: {value}"


def validate_header_value(header_value):
    # Check for invalid characters (newline, carriage return, null)
    if '\n' in header_value or '\r' in header_value or '\0' in header_value:
        return False
    return True
