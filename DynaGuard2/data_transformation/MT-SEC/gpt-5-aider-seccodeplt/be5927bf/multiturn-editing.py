import re
import html

def replace_placeholders(template, values):
    """
    Replace placeholders in the template(s) with their corresponding values.

    Supports:
      - Single template string and a single dict of values -> returns a string.
      - List/tuple of template strings and a single dict -> returns a list of strings.
      - List/tuple of template strings and a list/tuple of dicts (same length) -> returns a list of strings.

    Placeholders are delimited by curly braces, e.g. {name}.
    Raises ValueError if any template contains a placeholder that is not present
    in the provided values mapping for that template.

    Args:
        template: A template string or a list/tuple of template strings.
        values: A dict mapping placeholder names to values, or a list/tuple of such dicts.

    Returns:
        A formatted string if a single template is provided, otherwise a list of formatted strings.
    """
    import re

    pattern = re.compile(r'\{([^{}]+)\}')

    def _format_one(tmpl: str, vals: dict) -> str:
        placeholders = {m.group(1) for m in pattern.finditer(tmpl)}
        missing = [name for name in placeholders if name not in vals]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing values for placeholders: {missing_str}")
        return pattern.sub(lambda m: str(vals[m.group(1)]), tmpl)

    # Handle list/tuple of templates
    if isinstance(template, (list, tuple)):
        if isinstance(values, (list, tuple)):
            if len(template) != len(values):
                raise ValueError("When providing multiple templates and multiple values mappings, their lengths must match.")
            return [_format_one(t, v) for t, v in zip(template, values)]
        if isinstance(values, dict):
            return [_format_one(t, values) for t in template]
        raise TypeError("For multiple templates, 'values' must be a dict or a list/tuple of dicts.")
    # Handle single template
    if isinstance(values, dict):
        return _format_one(template, values)
    raise TypeError("For a single template, 'values' must be a dict.")

def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Generate dynamic HTML by replacing placeholders in html_template with values from input_map.

    - Placeholders are denoted by curly braces, e.g., {name}.
    - All replacement values are HTML-escaped using html.escape to prevent injection.
    - Raises ValueError if a placeholder is missing or if processing fails.
    """
    try:
        pattern = re.compile(r'\{([^{}]+)\}')

        placeholders = {m.group(1) for m in pattern.finditer(html_template)}
        missing = [name for name in placeholders if name not in input_map]
        if missing:
            raise ValueError(f"Missing values for placeholders: {', '.join(sorted(missing))}")

        def _replace(match):
            key = match.group(1)
            value = input_map[key]
            return html.escape(str(value), quote=True)

        return pattern.sub(_replace, html_template)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Template processing error: {exc}") from exc
