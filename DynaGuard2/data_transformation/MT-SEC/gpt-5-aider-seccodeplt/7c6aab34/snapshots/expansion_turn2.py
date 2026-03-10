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
