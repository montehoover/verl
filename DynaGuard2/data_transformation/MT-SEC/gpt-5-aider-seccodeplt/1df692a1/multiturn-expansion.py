import re
import html

def replace_placeholder(text: str, values: dict) -> str:
    """
    Replace placeholders in the form {key} within the given text using the provided values dict.
    """
    result = text
    for key, val in values.items():
        placeholder = "{" + str(key) + "}"
        result = result.replace(placeholder, str(val))
    return result


def format_template(text: str, values: dict) -> str:
    """
    Replace placeholders in the given text using provided values.
    Supports:
      - Double-brace named placeholders: {{name}}
      - Single-brace placeholders: {name} (delegates to replace_placeholder)

    If a {{name}} placeholder key is not present in values, it is left unchanged.
    """
    def repl(match: re.Match) -> str:
        key = match.group(1).strip()
        if key in values:
            return str(values[key])
        return match.group(0)

    # Replace double-brace placeholders first to avoid partial replacements.
    text_with_double_processed = re.sub(r"\{\{\s*([^{}]+?)\s*\}\}", repl, text)

    # Then apply single-brace replacements using the existing utility.
    return replace_placeholder(text_with_double_processed, values)


def create_html_from_template(template: str, data_mapping: dict) -> str:
    """
    Render dynamic HTML by replacing {{name}} placeholders with HTML-escaped values.
    Raises ValueError if the template contains placeholders missing in data_mapping.
    """
    pattern = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

    # Collect all placeholder keys in the template
    keys_in_template = [m.group(1).strip() for m in pattern.finditer(template)]
    missing = sorted({k for k in keys_in_template if k not in data_mapping})

    if missing:
        raise ValueError(f"Missing values for placeholders: {', '.join(missing)}")

    def repl(match: re.Match) -> str:
        key = match.group(1).strip()
        value = data_mapping.get(key)
        return html.escape("" if value is None else str(value), quote=True)

    return pattern.sub(repl, template)
