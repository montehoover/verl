import re
import html
from typing import Any, Mapping


def build_html_content(html_template: str, values_dict: Mapping[str, Any]) -> str:
    """
    Generate dynamic HTML by replacing {{...}} placeholders in the template
    with HTML-escaped values from the provided mapping.

    - Placeholders may include surrounding whitespace (e.g., {{ name }}).
    - Every placeholder found in the template must exist in the mapping; otherwise, ValueError is raised.
    - Replacement values are converted to strings and HTML-escaped (including quotes).

    Args:
        html_template: The HTML template containing placeholders like {{name}}.
        values_dict: A mapping from placeholder names to replacement values.

    Returns:
        The HTML string with all placeholders replaced.

    Raises:
        TypeError: If html_template is not a string or values_dict is not a mapping.
        ValueError: If a placeholder is not found in the provided mapping.
    """
    if not isinstance(html_template, str):
        raise TypeError("html_template must be a string")
    if not isinstance(values_dict, Mapping):
        raise TypeError("values_dict must be a mapping")

    pattern = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        if key not in values_dict:
            raise ValueError(f"Missing value for placeholder '{key}'")
        return html.escape(str(values_dict[key]), quote=True)

    return pattern.sub(replace, html_template)
