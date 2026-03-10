import re
import html

# Precompiled regex to find placeholders like {{ name }} with optional surrounding whitespace.
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][\w\-]*)\s*\}\}")

def create_html_from_template(template: str, data_mapping: dict) -> str:
    """
    Create HTML by mapping user-provided values to placeholders within a template.

    Args:
        template (str): An HTML template where placeholders exist as {{placeholder}}.
        data_mapping (dict): Mapping of placeholder names to their values.

    Returns:
        str: HTML content after processing the template and user data.

    Raises:
        ValueError: If template or data_mapping are of incorrect types, if the
                    placeholder mapping is incomplete, or if processing fails.
    """
    if not isinstance(template, str):
        raise ValueError("template must be a string")
    if not isinstance(data_mapping, dict):
        raise ValueError("data_mapping must be a dict")

    # Find all placeholders in the template
    placeholders = {m.group(1) for m in _PLACEHOLDER_RE.finditer(template)}

    # If there are no placeholders, return the template unchanged
    if not placeholders:
        return template

    # Ensure all placeholders are provided in the mapping
    missing = sorted(name for name in placeholders if name not in data_mapping)
    if missing:
        raise ValueError(f"Incomplete placeholder mapping. Missing keys: {', '.join(missing)}")

    # Replacement function that escapes values for safe HTML output
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        value = data_mapping.get(key)
        # Convert to string and escape for HTML safety
        return html.escape("" if value is None else str(value), quote=True)

    try:
        result = _PLACEHOLDER_RE.sub(_replace, template)
    except Exception as exc:
        raise ValueError(f"Failed to process template: {exc}") from exc

    return result
