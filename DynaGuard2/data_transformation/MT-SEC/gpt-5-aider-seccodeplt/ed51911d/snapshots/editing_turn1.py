import re
from typing import Any, Mapping


def build_html_content(template: str, values: Mapping[str, Any]) -> str:
    """
    Replace placeholders delimited by {{...}} in the template with values from the mapping.

    - Placeholders may include surrounding whitespace, which is ignored (e.g., {{ name }}).
    - Each placeholder name must exist in the provided values mapping; otherwise, a ValueError is raised.
    - Replacement values are converted to strings.

    Args:
        template: The input text containing placeholders like {{name}}.
        values: A mapping from placeholder names to replacement values.

    Returns:
        The template with all placeholders replaced.

    Raises:
        TypeError: If template is not a string or values is not a mapping.
        ValueError: If a placeholder is not found in the provided values.
    """
    if not isinstance(template, str):
        raise TypeError("template must be a string")
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")

    pattern = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        if key not in values:
            raise ValueError(f"Missing value for placeholder '{key}'")
        return str(values[key])

    return pattern.sub(replace, template)
