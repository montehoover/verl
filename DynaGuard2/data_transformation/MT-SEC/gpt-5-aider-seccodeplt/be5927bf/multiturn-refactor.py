"""
A small utility module for safely rendering HTML templates.

This module provides:
- A compiled regular expression pattern to detect placeholders in templates.
- Pure helper functions to extract placeholders and to HTML-escape values.
- A main function, `render_html_template`, that replaces placeholders with
  user-provided values after escaping them for safe HTML rendering.

Logging:
- Uses the module-level logger to trace template rendering operations.
- Logs the input template, detected placeholders, and the final rendered HTML.
- Logs missing placeholders at error level.

Note: This module does not configure logging. To see logs, configure logging
in the consuming application, e.g.:

    import logging
    logging.basicConfig(level=logging.DEBUG)
"""

import logging
import re
import html
from typing import Set

# Module-level logger
logger = logging.getLogger(__name__)

# Compiled regex pattern for placeholders like {{ name }}
PLACEHOLDER_PATTERN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_\-\.]*)\s*}}")


def extract_placeholders(template: str) -> Set[str]:
    """
    Extract placeholder names from the given template.

    Placeholders follow the pattern defined by PLACEHOLDER_PATTERN and must:
      - Start with a letter or underscore.
      - Contain letters, digits, underscores, hyphens, or dots thereafter.

    Args:
        template: The template string to analyze.

    Returns:
        A set of placeholder names found within the template.

    Example:
        >>> extract_placeholders('Hello, {{ user.name }}!')
        {'user.name'}
    """
    placeholders = {m.group(1) for m in PLACEHOLDER_PATTERN.finditer(template)}
    logger.debug("extract_placeholders: found placeholders=%s", sorted(placeholders))
    return placeholders


def escape_html_value(value) -> str:
    """
    Convert a value to string and escape it for safe inclusion in HTML.

    - None is treated as an empty string.
    - The resulting string is escaped using html.escape with quote=True.

    Args:
        value: Any value that can be converted to a string.

    Returns:
        A safely escaped HTML string.

    Raises:
        ValueError: If the value cannot be converted to a string.

    Example:
        >>> escape_html_value('<script>alert(1)</script>')
        '&lt;script&gt;alert(1)&lt;/script&gt;'
    """
    if value is None:
        return ""
    try:
        value_str = str(value)
    except Exception as exc:  # noqa: BLE001 (explicitly capturing to wrap)
        raise ValueError(f"Error converting value to string: {exc}") from exc
    escaped = html.escape(value_str, quote=True)
    logger.debug("escape_html_value: original=%r escaped=%r", value_str, escaped)
    return escaped


def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Render an HTML template by replacing placeholders with escaped values.

    Placeholders use the format: {{ placeholder_name }}
    Allowed placeholder names:
      - Start with a letter or underscore.
      - May include letters, digits, underscores, hyphens, and dots.

    Args:
        html_template: The HTML template string containing placeholders.
        input_map: A dictionary mapping placeholder names to their values.

    Returns:
        The rendered HTML string with placeholders replaced by safely escaped values.

    Raises:
        ValueError: If:
            - html_template is not a string.
            - input_map is not a dict.
            - A placeholder in the template is missing from input_map.
            - Processing fails for any unforeseen reason (wrapped as ValueError).

    Logging:
        - Logs the input template at DEBUG level.
        - Logs the set of detected placeholders at DEBUG level.
        - Logs missing placeholders at ERROR level.
        - Logs each placeholder replacement at DEBUG level.
        - Logs the final rendered HTML at DEBUG level.
    """
    if not isinstance(html_template, str):
        raise ValueError("html_template must be a string")
    if not isinstance(input_map, dict):
        raise ValueError("input_map must be a dict")

    logger.debug("render_html_template: input_template=%s", html_template)

    try:
        # Validate all placeholders are present
        placeholders_in_template = extract_placeholders(html_template)
        missing = [name for name in placeholders_in_template if name not in input_map]
        if missing:
            missing_sorted = ", ".join(sorted(missing))
            logger.error(
                "render_html_template: missing placeholders: %s", missing_sorted
            )
            raise ValueError(f"Missing values for placeholders: {missing_sorted}")

        def _replace(match: re.Match) -> str:
            key = match.group(1)
            if key not in input_map:
                # Redundant due to pre-check but kept for safety
                logger.error(
                    "render_html_template: missing value during substitution for key=%s",
                    key,
                )
                raise ValueError(f"Missing value for placeholder '{key}'")
            logger.debug("render_html_template: replacing key=%s", key)
            try:
                return escape_html_value(input_map.get(key))
            except ValueError as exc:
                raise ValueError(
                    f"Error converting value for placeholder '{key}': {exc}"
                ) from exc

        # Perform replacement
        result = PLACEHOLDER_PATTERN.sub(_replace, html_template)
        logger.debug("render_html_template: rendered_html=%s", result)
        return result

    except ValueError:
        # Re-raise controlled errors intact
        raise
    except Exception as exc:
        # Wrap any unexpected errors into a ValueError as specified
        raise ValueError(f"Error processing template: {exc}") from exc
