import re

def generate_command():
    return "SELECT * FROM users;"

def generate_command_with_header(value):
    return f"SELECT {value} FROM users;"

def include_custom_header(custom_value: str) -> str:
    """
    Build a custom HTTP header line with a sanitized header value.

    Sanitization removes CR/LF to prevent header injection, strips other control
    characters (except horizontal tab), collapses whitespace, and trims edges.

    May raise:
        re.error: If the underlying regular expression operations fail.
    """
    value = str(custom_value)

    # Prevent header injection via CRLF
    value = re.sub(r'[\r\n]+', ' ', value)

    # Remove other control chars except HT (0x09)
    value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)

    # Normalize whitespace (spaces/tabs) and trim
    value = re.sub(r'[ \t]+', ' ', value).strip()

    return f"X-Custom-Header: {value}"
