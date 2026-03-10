import re
import html
from typing import Dict, Any

def generate_dynamic_html(template: str, user_input: Dict[str, Any]) -> str:
    """
    Generate dynamic HTML by replacing {{placeholder}} tokens in the template with values
    provided in user_input. Values are HTML-escaped.

    Args:
        template: An HTML template string containing placeholders like {{ name }}.
        user_input: A mapping from placeholder names to values.

    Returns:
        The generated HTML string with placeholders replaced.

    Raises:
        ValueError: If the template is invalid (unbalanced braces, malformed placeholders),
                    if a required placeholder is missing in user_input, or on processing errors.
    """
    # Validate input types
    if not isinstance(template, str) or not isinstance(user_input, dict):
        raise ValueError("Invalid input: template must be str and user_input must be dict")

    # Quick path: empty template
    if template == "":
        return ""

    # Validate template structure for placeholders
    opens = template.count("{{")
    closes = template.count("}}")
    if opens != closes:
        raise ValueError("Invalid template: unmatched placeholder braces")

    # If there are no placeholders, return the template as-is
    if opens == 0:
        return template

    # Match only well-formed placeholders of the form {{ name }}
    strong_pattern = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

    # Ensure every {{ is part of a well-formed placeholder
    strong_matches = list(strong_pattern.finditer(template))
    if len(strong_matches) != opens:
        # Detect malformed placeholders like {{ 123 }}, {{a-b}}, nested braces, etc.
        raise ValueError("Invalid template: malformed placeholder(s)")

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in user_input:
            raise ValueError(f"Missing value for placeholder: {key}")
        value = user_input[key]
        # Convert to string and HTML-escape
        escaped = html.escape(str(value), quote=True)
        return escaped

    try:
        result = strong_pattern.sub(replace, template)
    except Exception as exc:
        # Normalize any unexpected error into ValueError as per contract
        raise ValueError(f"Template processing failed: {exc}") from exc

    return result
