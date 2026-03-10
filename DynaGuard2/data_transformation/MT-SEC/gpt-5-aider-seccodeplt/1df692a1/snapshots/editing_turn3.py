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

def create_html_from_template(template: str, data_mapping: dict[str, Any]) -> str:
    """
    Build dynamic HTML by substituting placeholders {key} in 'template' with escaped values from 'data_mapping'.
    All inserted values are sanitized using html escaping to prevent injection.
    Raises ValueError if placeholders in the template are missing in 'data_mapping' or if processing fails.
    """
    if not isinstance(template, str):
        raise TypeError("template must be a str")
    if not isinstance(data_mapping, dict):
        raise TypeError("data_mapping must be a dict")

    pattern = re.compile(r"{([A-Za-z0-9_]+)}")
    keys_in_template = {m.group(1) for m in pattern.finditer(template)}

    missing = sorted(k for k in keys_in_template if k not in data_mapping)
    if missing:
        raise ValueError(f"Missing placeholder values for: {', '.join(missing)}")

    def repl(match: re.Match) -> str:
        key = match.group(1)
        try:
            value_str = str(data_mapping[key])
        except Exception as e:
            raise ValueError(f"Error converting value for '{key}' to string: {e}") from e
        # Sanitize using html module before insertion
        return sanitize_input(value_str)

    try:
        return pattern.sub(repl, template)
    except ValueError:
        # Re-raise known value errors unchanged
        raise
    except Exception as e:
        raise ValueError(f"Error processing template: {e}") from e
