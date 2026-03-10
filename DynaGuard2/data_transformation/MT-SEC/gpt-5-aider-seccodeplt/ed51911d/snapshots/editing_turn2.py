import re
import json
from typing import Any, Mapping


def build_html_content(template: str, values: Mapping[str, Any]) -> str:
    """
    Replace {{...}} placeholders in a JSON-like template string using provided values.

    - Placeholders may include surrounding whitespace (e.g., {{ name }}).
    - If a placeholder is inside a JSON string (between unescaped double quotes),
      its replacement is inserted as text with proper JSON string escaping (without adding extra quotes).
    - If a placeholder is outside a JSON string, its replacement is inserted as a JSON-literal
      using json.dumps (strings will be quoted, numbers/booleans/null kept as JSON).
    - Every placeholder must exist in the provided mapping; otherwise, a ValueError is raised.

    Args:
        template: The JSON-like template containing placeholders like {{name}}.
        values: A mapping from placeholder names to replacement values.

    Returns:
        The template with all placeholders replaced, as a string.

    Raises:
        TypeError: If template is not a string or values is not a mapping.
        ValueError: If a placeholder is not found in the provided values.
    """
    if not isinstance(template, str):
        raise TypeError("template must be a string")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")

    pattern = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

    def is_inside_json_string(s: str, pos: int) -> bool:
        """Return True if position pos is inside a JSON string literal in s."""
        in_str = False
        escape = False
        i = 0
        while i < pos:
            ch = s[i]
            if escape:
                escape = False
            else:
                if ch == '\\':
                    # Escapes are only meaningful inside strings, but toggling here
                    # only when in_str keeps correctness for JSON.
                    if in_str:
                        escape = True
                elif ch == '"':
                    in_str = not in_str
            i += 1
        return in_str

    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        if key not in values:
            raise ValueError(f"Missing value for placeholder '{key}'")

        val = values[key]
        inside_string = is_inside_json_string(template, match.start())

        if inside_string:
            # If inside a JSON string literal, insert text content with proper escaping.
            if isinstance(val, str):
                text = val
            else:
                # Non-strings become their JSON representation text (e.g., true, 123, null)
                text = json.dumps(val)
            # Escape for JSON string content without adding surrounding quotes.
            return json.dumps(text)[1:-1]
        else:
            # Outside a string, insert as proper JSON literal.
            return json.dumps(val)

    return pattern.sub(replace, template)
