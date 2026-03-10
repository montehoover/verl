import re
import html
from typing import Dict, Any, List, Match

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

def parse_placeholders(template: str) -> List[Match[str]]:
    """
    Parse and validate placeholders in the template.

    Returns a list of regex Match objects for each placeholder occurrence.

    Raises:
        ValueError: if braces are unmatched or placeholders are malformed.
    """
    opens = template.count("{{")
    closes = template.count("}}")
    if opens != closes:
        raise ValueError("Invalid template: unmatched placeholder braces")

    if opens == 0:
        return []

    matches = list(PLACEHOLDER_PATTERN.finditer(template))
    if len(matches) != opens:
        raise ValueError("Invalid template: malformed placeholder(s)")

    return matches

def replace_placeholders(template: str, user_input: Dict[str, Any]) -> str:
    """
    Replace placeholders in the template with escaped values from user_input.

    Raises:
        ValueError: if a required placeholder is missing in user_input.
    """
    def _repl(m: Match[str]) -> str:
        key = m.group(1)
        if key not in user_input:
            raise ValueError(f"Missing value for placeholder: {key}")
        return html.escape(str(user_input[key]), quote=True)

    return PLACEHOLDER_PATTERN.sub(_repl, template)

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
    if not isinstance(template, str) or not isinstance(user_input, dict):
        raise ValueError("Invalid input: template must be str and user_input must be dict")

    if template == "":
        return ""

    # Quick return if no placeholders at all
    if "{{" not in template and "}}" not in template:
        return template

    # Parse and validate placeholders
    parse_placeholders(template)

    try:
        return replace_placeholders(template, user_input)
    except ValueError:
        # Re-raise known validation errors unchanged
        raise
    except Exception as exc:
        raise ValueError(f"Template processing failed: {exc}") from exc
