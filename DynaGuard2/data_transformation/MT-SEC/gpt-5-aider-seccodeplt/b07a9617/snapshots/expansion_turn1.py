import re
from typing import Any


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
