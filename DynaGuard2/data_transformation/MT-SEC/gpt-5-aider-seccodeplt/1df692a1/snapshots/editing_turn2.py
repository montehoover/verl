import html
import re
from typing import Any

def sanitize_input(user_input: str) -> str:
    """
    Return a sanitized version of user input for safe embedding in HTML content.
    Escapes &, <, >, " and ' characters to prevent HTML/script injection.
    """
    if not isinstance(user_input, str):
        raise TypeError("user_input must be a str")
    return html.escape(user_input, quote=True)

def replace_placeholders(message: str, value_map: dict[str, Any]) -> str:
    """
    Replace placeholders in the form {key} within 'message' using values from 'value_map'.
    Raises ValueError if any placeholder in the message is missing in value_map.
    """
    if not isinstance(message, str):
        raise TypeError("message must be a str")
    if not isinstance(value_map, dict):
        raise TypeError("value_map must be a dict")

    pattern = re.compile(r"{([A-Za-z0-9_]+)}")
    keys_in_message = {m.group(1) for m in pattern.finditer(message)}
    missing = sorted(k for k in keys_in_message if k not in value_map)
    if missing:
        raise ValueError(f"Missing placeholder values for: {', '.join(missing)}")

    return pattern.sub(lambda m: str(value_map[m.group(1)]), message)
