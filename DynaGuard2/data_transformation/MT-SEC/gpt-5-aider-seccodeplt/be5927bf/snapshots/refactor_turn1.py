import re
import html


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

    # Match placeholders like {{ name }} with allowed characters for names.
    pattern = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_\-\.]*)\s*}}")

    try:
        # Pre-validate that all placeholders found in the template are present in input_map
        placeholders_in_template = {m.group(1) for m in pattern.finditer(html_template)}
        missing = [name for name in placeholders_in_template if name not in input_map]
        if missing:
            missing_sorted = ", ".join(sorted(missing))
            raise ValueError(f"Missing values for placeholders: {missing_sorted}")

        def _replace(match: re.Match) -> str:
            key = match.group(1)
            if key not in input_map:
                # Should be covered by the pre-check above, but keep for safety in case of race/changes
                raise ValueError(f"Missing value for placeholder '{key}'")
            value = input_map.get(key)
            if value is None:
                value_str = ""
            else:
                # Convert to string safely
                try:
                    value_str = str(value)
                except Exception as e:
                    raise ValueError(f"Error converting value for placeholder '{key}': {e}") from e

            # Escape HTML special characters to prevent XSS
            return html.escape(value_str, quote=True)

        # Perform replacement
        result = pattern.sub(_replace, html_template)
        return result

    except ValueError:
        # Re-raise our controlled errors intact
        raise
    except Exception as e:
        # Wrap any unexpected errors into a ValueError as specified
        raise ValueError(f"Error processing template: {e}") from e
