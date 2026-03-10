import re
import html
from typing import Any, Dict

# Matches {placeholder} where placeholder is an identifier with optional dot paths,
# and ignores double-braced sequences like {{placeholder}}.
_PLACEHOLDER_RE = re.compile(r'(?<!{)\{([A-Za-z_][A-Za-z0-9_\.]*)\}(?!})')

_MISSING = object()


def _lookup_value(path: str, values: Dict[str, Any]) -> Any:
    """
    Look up a possibly dotted path (e.g., "user.name") within a nested dict.
    Returns _MISSING if any segment is absent.
    """
    current: Any = values
    for part in path.split('.'):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def replace_placeholders(text: str, values: Dict[str, Any]) -> str:
    """
    Replace placeholders in the given text using the provided values.

    Placeholder syntax:
      - {key} for simple keys
      - {a.b.c} for nested dict lookup along dot-separated path

    Behavior:
      - If a placeholder's value is missing, the placeholder is left unchanged.
      - Values are converted to strings via str() before substitution.
      - Double-braced sequences like {{key}} are left untouched.

    Example:
      text = "Hello {user.name}, your order {order.id} is {order.status}."
      values = {"user": {"name": "Ada"}, "order": {"id": 42, "status": "ready"}}
      -> "Hello Ada, your order 42 is ready."
    """
    if not text:
        return text

    def _replacer(match: re.Match) -> str:
        keypath = match.group(1)
        value = _lookup_value(keypath, values)
        if value is _MISSING:
            return match.group(0)  # leave placeholder as-is
        return str(value)

    return _PLACEHOLDER_RE.sub(_replacer, text)


def escape_html_content(text: str) -> str:
    """
    Escape HTML special characters in the given text to make it safe for HTML rendering.
    This escapes &, <, >, ", and ' characters.
    """
    return html.escape(text, quote=True)


def _escape_values(obj: Any) -> Any:
    """
    Recursively escape values in a dict for safe HTML rendering.
    Non-dict values are converted to strings and escaped.
    """
    if isinstance(obj, dict):
        return {k: _escape_values(v) for k, v in obj.items()}
    return html.escape(str(obj), quote=True)


def generate_dynamic_html(template: str, user_input: Dict[str, Any]) -> str:
    """
    Generate dynamic HTML by replacing placeholders in the template with user-provided values.
    Values are HTML-escaped to prevent injection issues.

    Args:
        template: HTML template string containing placeholders like {name} or {user.name}.
        user_input: Dictionary with values for placeholders. Supports nested dicts for dotted paths.

    Returns:
        A string with placeholders replaced by escaped values.

    Raises:
        ValueError: If the template is not a string or cannot be processed.
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string.")
    if not isinstance(user_input, dict):
        raise ValueError("user_input must be a dictionary.")

    safe_values = _escape_values(user_input)
    try:
        return replace_placeholders(template, safe_values)
    except Exception as exc:
        raise ValueError("Template could not be processed.") from exc
