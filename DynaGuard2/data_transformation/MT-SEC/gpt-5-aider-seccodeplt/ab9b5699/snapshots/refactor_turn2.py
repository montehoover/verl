import re
import html

__all__ = ["create_html_template"]

# Matches placeholders like {{ name }}, capturing the inner content non-greedily.
_PLACEHOLDER_PATTERN = re.compile(r"{{\s*(.*?)\s*}}")


def _extract_placeholder_name(match: re.Match) -> str:
    """
    Extract and validate the placeholder name from a placeholder match.

    Args:
        match (re.Match): A regex match object for a placeholder.

    Returns:
        str: The validated placeholder name.

    Raises:
        ValueError: If the placeholder is empty or malformed.
    """
    try:
        inner = match.group(1)
    except IndexError as e:
        raise ValueError("Invalid placeholder format") from e

    name = (inner or "").strip()
    if not name:
        raise ValueError("Encountered empty placeholder {{}} in template")
    return name


def _insert_value(name: str, user_values: dict) -> str:
    """
    Resolve and escape the value for a given placeholder name.

    Args:
        name (str): The placeholder name to resolve.
        user_values (dict): Mapping from placeholder names to user-provided values.

    Returns:
        str: The HTML-escaped string value for insertion.

    Raises:
        ValueError: If the placeholder name is missing in user_values.
    """
    if name not in user_values:
        raise ValueError(f"Missing value for placeholder '{name}'")
    value = user_values[name]
    # Always escape user-provided values for safety.
    return html.escape("" if value is None else str(value), quote=True)


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
        def _replacer(match: re.Match) -> str:
            name = _extract_placeholder_name(match)
            return _insert_value(name, user_values)

        result = _PLACEHOLDER_PATTERN.sub(_replacer, html_template)
        return result
    except Exception as e:
        # Normalize all processing errors to ValueError as specified.
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Template processing failed: {e}") from e
