import re
import html

def build_html_content(html_template: str, values_dict: dict) -> str:
    """
    Build HTML content by replacing {{...}} placeholders in the given template
    with corresponding values from values_dict. Values are HTML-escaped.

    Args:
        html_template: An HTML template string containing placeholders delimited with {{...}}.
        values_dict: A mapping of placeholder names to values to be inserted.

    Returns:
        The rendered HTML string.

    Raises:
        ValueError: If input types are invalid, a placeholder is missing,
                    an invalid placeholder is encountered, or any error occurs during processing.
    """
    try:
        if not isinstance(html_template, str):
            raise ValueError("html_template must be a string")
        if not isinstance(values_dict, dict):
            raise ValueError("values_dict must be a dict")

        # Match {{ placeholder }} where placeholder cannot contain braces.
        pattern = re.compile(r"\{\{\s*([^\{\}]+?)\s*\}\}")

        # Find and normalize all placeholders
        raw_matches = pattern.findall(html_template)
        placeholders = [m.strip() for m in raw_matches]

        # Check for invalid (empty) placeholders like {{   }}
        invalid = [k for k in placeholders if k == ""]
        if invalid:
            raise ValueError("Invalid empty placeholder found in template")

        # Ensure all placeholders have corresponding values
        missing = sorted({k for k in placeholders if k not in values_dict})
        if missing:
            raise ValueError(f"Missing placeholder values for: {', '.join(missing)}")

        # Perform substitution with HTML-escaped values
        def _replace(match: re.Match) -> str:
            key = match.group(1).strip()
            value = values_dict[key]
            return html.escape(str(value), quote=True)

        return pattern.sub(_replace, html_template)

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error during template processing: {e}") from None
