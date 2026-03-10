import re
import html

# Pattern matches placeholders like {{ name }} with optional whitespace
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Render an HTML template by safely substituting placeholders with user-provided values.

    Args:
        html_template (str): The HTML template containing placeholders like {{placeholder}}.
        input_map (dict): A mapping from placeholder names to values to insert.

    Returns:
        str: The rendered HTML string with placeholders replaced.

    Raises:
        ValueError: If a placeholder in the template is missing from input_map,
                    or if processing the template encounters an error.
    """
    if not isinstance(html_template, str):
        raise ValueError("html_template must be a string")
    if not isinstance(input_map, dict):
        raise ValueError("input_map must be a dictionary")

    try:
        placeholders = _PLACEHOLDER_RE.findall(html_template)
        # Validate all placeholders exist in the provided input_map
        missing = [name for name in set(placeholders) if name not in input_map]
        if missing:
            raise ValueError(f"Missing value(s) for placeholder(s): {', '.join(sorted(missing))}")

        def _replace(match: re.Match) -> str:
            key = match.group(1)
            value = input_map[key]
            # Ensure safe HTML by escaping user-provided values
            return html.escape(str(value), quote=True)

        rendered = _PLACEHOLDER_RE.sub(_replace, html_template)
        return rendered
    except ValueError:
        # Re-raise explicit ValueErrors as-is
        raise
    except Exception as exc:
        # Wrap any unexpected error into a ValueError as required
        raise ValueError(f"Template processing error: {exc}")
