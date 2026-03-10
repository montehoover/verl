import re

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
