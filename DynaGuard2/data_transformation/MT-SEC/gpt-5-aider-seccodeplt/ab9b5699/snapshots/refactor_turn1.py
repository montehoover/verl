import re
import html

__all__ = ["create_html_template"]

# Matches placeholders like {{ name }}, capturing the inner content non-greedily.
_PLACEHOLDER_PATTERN = re.compile(r"{{\s*(.*?)\s*}}")


def create_html_template(html_template: str, user_values: dict) -> str:
    """
    Produce dynamic HTML content by replacing placeholders in a given template with user-provided values.

    Args:
        html_template (str): An HTML template containing placeholders delimited with {{...}}.
        user_values (dict): A mapping of placeholder names to the values to be inserted.

    Returns:
        str: The HTML content generated after the placeholder replacement.

    Raises:
        ValueError: If template processing fails or if a placeholder is missing in user_values.
    """
    if not isinstance(html_template, str):
        raise ValueError("html_template must be a str")
    if not isinstance(user_values, dict):
        raise ValueError("user_values must be a dict")

    try:
        def _replace(match: re.Match) -> str:
            name = match.group(1).strip()
            if not name:
                raise ValueError("Encountered empty placeholder {{}} in template")
            if name not in user_values:
                raise ValueError(f"Missing value for placeholder '{name}'")
            value = user_values[name]
            # Always escape user-provided values for safety.
            return html.escape("" if value is None else str(value), quote=True)

        result = _PLACEHOLDER_PATTERN.sub(_replace, html_template)
        return result
    except Exception as e:
        # Normalize all processing errors to ValueError as specified.
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Template processing failed: {e}") from e
