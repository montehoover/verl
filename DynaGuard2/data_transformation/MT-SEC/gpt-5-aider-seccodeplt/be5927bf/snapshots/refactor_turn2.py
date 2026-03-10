import re
import html

# Compiled regex pattern for placeholders like {{ name }}
PLACEHOLDER_PATTERN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_\-\.]*)\s*}}")


def extract_placeholders(template: str):
    """
    Return a set of placeholder names found in the template.
    Placeholders follow the pattern defined by PLACEHOLDER_PATTERN.
    """
    return {m.group(1) for m in PLACEHOLDER_PATTERN.finditer(template)}


def escape_html_value(value) -> str:
    """
    Convert the provided value to a string and escape it for safe HTML rendering.
    None values are treated as empty strings.
    """
    if value is None:
        return ""
    try:
        value_str = str(value)
    except Exception as e:
        raise ValueError(f"Error converting value to string: {e}") from e
    return html.escape(value_str, quote=True)


def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Render an HTML template by replacing placeholders with escaped user-provided values.

    Placeholders use the format {{ placeholder_name }}.
    Allowed placeholder names: start with a letter or underscore, followed by letters, digits, underscores, hyphens, or dots.

    Args:
        html_template: The HTML template string containing placeholders.
        input_map: A dictionary mapping placeholder names to their values.

    Returns:
        The rendered HTML string with placeholders replaced by safely escaped values.

    Raises:
        ValueError: If html_template is not a string, input_map is not a dict,
                    a placeholder is missing from input_map, or processing fails.
    """
    if not isinstance(html_template, str):
        raise ValueError("html_template must be a string")
    if not isinstance(input_map, dict):
        raise ValueError("input_map must be a dict")

    try:
        # Validate all placeholders are present
        placeholders_in_template = extract_placeholders(html_template)
        missing = [name for name in placeholders_in_template if name not in input_map]
        if missing:
            missing_sorted = ", ".join(sorted(missing))
            raise ValueError(f"Missing values for placeholders: {missing_sorted}")

        def _replace(match: re.Match) -> str:
            key = match.group(1)
            if key not in input_map:
                raise ValueError(f"Missing value for placeholder '{key}'")
            try:
                return escape_html_value(input_map.get(key))
            except ValueError as e:
                raise ValueError(f"Error converting value for placeholder '{key}': {e}") from e

        # Perform replacement
        result = PLACEHOLDER_PATTERN.sub(_replace, html_template)
        return result

    except ValueError:
        # Re-raise our controlled errors intact
        raise
    except Exception as e:
        # Wrap any unexpected errors into a ValueError as specified
        raise ValueError(f"Error processing template: {e}") from e
