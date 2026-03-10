import re
import html

PLACEHOLDER_REGEX = re.compile(r'{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}')

def extract_placeholders(template):
    if not template:
        return []
    return PLACEHOLDER_REGEX.findall(template)

def replace_placeholders(template, values):
    """
    Replace {{placeholder}} occurrences in template using values dict.
    Values are HTML-escaped to prevent injection.
    Placeholders without a corresponding key are left unchanged.
    """
    if template is None:
        return ''
    if not isinstance(template, str):
        template = str(template)

    def repl(match):
        key = match.group(1)
        if key in values:
            val = '' if values[key] is None else str(values[key])
            return html.escape(val, quote=True)
        return match.group(0)

    return PLACEHOLDER_REGEX.sub(repl, template)

def render_html_template(html_template, input_map):
    """
    Render an HTML template by replacing {{placeholder}} tokens with values from input_map.
    - html_template: str containing the HTML with placeholders.
    - input_map: dict mapping placeholder names to values.
    Returns the rendered HTML string.
    Raises ValueError if:
      - html_template is None or not coercible to string,
      - input_map is not a dict,
      - any placeholder in the template is missing from input_map,
      - any other processing error occurs.
    """
    try:
        if html_template is None:
            raise ValueError("html_template must be a string")
        if not isinstance(html_template, str):
            html_template = str(html_template)
        if not isinstance(input_map, dict):
            raise ValueError("input_map must be a dict")

        placeholders = extract_placeholders(html_template)
        if placeholders:
            missing = sorted(set(p for p in placeholders if p not in input_map))
            if missing:
                raise ValueError(f"Missing values for placeholders: {', '.join(missing)}")

        rendered = replace_placeholders(html_template, input_map)

        # Ensure no unresolved placeholders remain
        remaining = PLACEHOLDER_REGEX.findall(rendered)
        if remaining:
            raise ValueError(f"Unresolved placeholders after rendering: {', '.join(sorted(set(remaining)))}")

        return rendered
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error rendering HTML template: {e}")
