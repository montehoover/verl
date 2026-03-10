import re
import html

# Precompiled regex for placeholders of the form {{ name }}
_PLACEHOLDER_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def generate_dynamic_html(template: str, user_input: dict) -> str:
    """
    Generate dynamic HTML by replacing {{placeholders}} in the template with values from user_input.

    Args:
        template: str - An HTML template containing placeholders like {{name}}.
        user_input: dict - Mapping from placeholder names to their values.

    Returns:
        str - The generated HTML content with placeholders replaced.

    Raises:
        ValueError:
            - If template is not a string or user_input is not a dict.
            - If the template contains malformed or unresolved placeholders.
            - If a placeholder in the template is missing from user_input.
            - If a value cannot be converted to string.
    """
    if not isinstance(template, str):
        raise ValueError("Invalid template: expected a string.")
    if not isinstance(user_input, dict):
        raise ValueError("Invalid user_input: expected a dict.")

    # Find all placeholders in the template.
    placeholders = [m.group(1) for m in _PLACEHOLDER_RE.finditer(template)]

    # If there are any double-brace markers but none match the valid placeholder pattern,
    # treat it as malformed template.
    if (("{{" in template) or ("}}" in template)) and not placeholders:
        raise ValueError("Invalid template: malformed placeholder(s).")

    # Ensure all placeholders have provided values.
    missing = sorted({name for name in set(placeholders) if name not in user_input})
    if missing:
        raise ValueError(f"Missing values for placeholders: {', '.join(missing)}")

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        value = user_input.get(key)
        try:
            text = "" if value is None else str(value)
        except Exception as e:
            raise ValueError(f"Cannot convert value for '{key}' to string: {e}") from e
        # Escape HTML to prevent injection; quotes are escaped as well.
        return html.escape(text, quote=True)

    try:
        result = _PLACEHOLDER_RE.sub(_replace, template)
    except Exception as e:
        raise ValueError(f"Failed to process template: {e}") from e

    # After replacement, any remaining '{{' or '}}' indicates unresolved or malformed placeholders.
    if ("{{" in result) or ("}}" in result):
        raise ValueError("Invalid template: unresolved or malformed placeholders remain after processing.")

    return result
