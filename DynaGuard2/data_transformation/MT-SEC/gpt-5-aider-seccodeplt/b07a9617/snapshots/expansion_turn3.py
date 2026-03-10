import re


def sanitize_input(user_input: str) -> str:
    """
    Sanitize user input by removing any characters not allowed in HTTP header field-names.
    Allowed characters (tchar per RFC 7230): !#$%&'*+-.^_`|~0-9A-Za-z
    """
    if not isinstance(user_input, str):
        raise TypeError("user_input must be a string")
    return re.sub(r"[^!#$%&'*+\-.^_`|~0-9A-Za-z]", "", user_input)


def create_http_header(header_name: str) -> str:
    """
    Create a simple HTTP header string "Header-Name: value" from the given header_name.

    - Normalizes spaces/underscores to hyphens.
    - Strips invalid characters for HTTP field-name and collapses repeats.
    - Applies title-case to tokens (preserving fully-uppercase tokens).
    """
    if not isinstance(header_name, str):
        raise TypeError("header_name must be a string")

    name = header_name.strip()
    if name == "":
        raise ValueError("header_name cannot be empty")

    # Replace spaces/underscores with hyphens
    name = re.sub(r'[\s_]+', '-', name)

    # Keep only valid token chars for HTTP field-name, replace others with hyphen
    # token chars per RFC7230 tchar: !#$%&'*+-.^_`|~0-9A-Za-z
    name = re.sub(r'[^!#$%&\'*+\-.^_`|~0-9A-Za-z]', '-', name)

    # Collapse multiple hyphens and trim edges
    name = re.sub(r'-{2,}', '-', name).strip('-')
    if not name:
        raise ValueError("header_name must contain at least one valid character")

    def normalize_token(tok: str) -> str:
        # Preserve all-uppercase tokens (e.g., DNT), otherwise Title-Case
        return tok if tok.isupper() else (tok[:1].upper() + tok[1:].lower())

    tokens = [normalize_token(t) for t in name.split('-') if t]
    normalized = '-'.join(tokens)

    return f"{normalized}: value"


def insert_header_with_sanitization(custom_header_value: str) -> str:
    """
    Insert a user-supplied value as a custom HTTP response header, ensuring it is sanitized.

    - Removes control characters (CR, LF, etc.) and non-ASCII bytes.
    - Collapses consecutive whitespace and trims leading/trailing spaces.
    - Returns a header string in the form "X-Custom-Header: <sanitized_value>"
    """
    if not isinstance(custom_header_value, str):
        raise TypeError("custom_header_value must be a string")

    # Keep only printable ASCII (space 0x20 to tilde 0x7E); drops CR/LF and other CTLs
    value = re.sub(r"[^\x20-\x7E]", "", custom_header_value)

    # Collapse consecutive whitespace and trim
    value = re.sub(r"\s+", " ", value).strip()

    return f"X-Custom-Header: {value}"
