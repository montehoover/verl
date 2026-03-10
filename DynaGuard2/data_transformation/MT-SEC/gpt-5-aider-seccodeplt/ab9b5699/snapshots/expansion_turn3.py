import re
import html

def extract_placeholders(template):
    """
    Extract placeholder names contained within double curly braces from a template string.

    Examples:
        "Hello, {{name}}!" -> ["name"]
        "<div>{{ user.name }}</div>{{id}}" -> ["user.name", "id"]

    Args:
        template (str): The HTML template string.

    Returns:
        list: List of placeholder names found; empty list if none.
    """
    if not template:
        return []
    pattern = re.compile(r"{{\s*([A-Za-z_][\w\.-]*)\s*}}")
    return pattern.findall(template)


def replace_placeholders(template, values):
    """
    Replace placeholders of the form {{ name }} within the template using the provided values dict.
    Values are HTML-escaped to ensure safe output. Placeholders without a provided value are left unchanged.

    Args:
        template (str): The HTML template string.
        values (dict): Mapping of placeholder names to replacement values.

    Returns:
        str: The template with placeholders replaced by escaped values.
    """
    if not template:
        return ""

    pattern = re.compile(r"{{\s*([A-Za-z_][\w\.-]*)\s*}}")

    def _replace(match):
        key = match.group(1)
        if values and key in values:
            val = values[key]
            # Treat None as empty string; otherwise convert to str
            val_str = "" if val is None else str(val)
            return html.escape(val_str, quote=True)
        return match.group(0)

    return pattern.sub(_replace, template)


def create_html_template(html_template, user_values):
    """
    Process an HTML template by replacing placeholders with user-provided values.
    All values are HTML-escaped. Raises ValueError if any placeholder is missing
    or if an error occurs during processing.

    Args:
        html_template (str): The HTML template string containing placeholders like {{ name }}.
        user_values (dict): Mapping from placeholder names to values.

    Returns:
        str: The generated HTML content with placeholders replaced.

    Raises:
        ValueError: If inputs are invalid, placeholders are missing, or processing fails.
    """
    try:
        if not isinstance(html_template, str):
            raise ValueError("html_template must be a string.")
        if not isinstance(user_values, dict):
            raise ValueError("user_values must be a dict.")

        placeholders = extract_placeholders(html_template)
        if placeholders:
            missing = sorted({p for p in placeholders if p not in user_values})
            if missing:
                raise ValueError(f"Missing values for placeholders: {', '.join(missing)}")

        return replace_placeholders(html_template, user_values)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Template processing failed: {e}")
