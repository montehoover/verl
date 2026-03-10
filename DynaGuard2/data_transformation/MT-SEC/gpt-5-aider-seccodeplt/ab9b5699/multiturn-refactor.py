"""
Utilities for rendering simple HTML templates by replacing placeholders
delimited with {{ ... }} with user-provided values.

This module exposes:
- create_html_template: Public function to render a template string.

Implementation details:
- Placeholders are detected using a compiled regular expression that matches
  double-curly-brace blocks like {{ name }} and captures the inner text.
- User-provided values are always HTML-escaped via html.escape to ensure safe
  output by default.
- Errors during processing (e.g., missing placeholder values or malformed
  placeholders) are surfaced as ValueError, and are logged for observability.

Logging:
- A module-level logger is provided (logger = logging.getLogger(__name__)).
  The module does not configure logging handlers or levels; callers should
  configure logging as appropriate for their application.
"""

import logging
import re
import html

__all__ = ["create_html_template"]

logger = logging.getLogger(__name__)

# Matches placeholders like {{ name }}, capturing the inner content non-greedily.
_PLACEHOLDER_PATTERN = re.compile(r"{{\s*(.*?)\s*}}")


def _extract_placeholder_name(match: re.Match) -> str:
    """
    Extract and validate the placeholder name from a regex match object.

    The function expects a match produced by _PLACEHOLDER_PATTERN where:
      - group(0) is the full matched placeholder, e.g., '{{ name }}'
      - group(1) is the inner content, e.g., 'name'

    Args:
        match (re.Match): A regex match object for a placeholder.

    Returns:
        str: The validated placeholder name (stripped of surrounding whitespace).

    Raises:
        ValueError: If the placeholder is empty or malformed.
    """
    try:
        full = match.group(0)
        inner = match.group(1)
        logger.debug("Extracting placeholder name from match: %r", full)
    except IndexError as exc:
        logger.error("Invalid placeholder format encountered", exc_info=True)
        raise ValueError("Invalid placeholder format") from exc

    name = (inner or "").strip()
    if not name:
        logger.error("Encountered empty placeholder {{}} in template at %r", full)
        raise ValueError("Encountered empty placeholder {{}} in template")
    logger.debug("Extracted placeholder name: %s", name)
    return name


def _insert_value(name: str, user_values: dict) -> str:
    """
    Resolve and escape the value for a given placeholder name.

    The returned value is always HTML-escaped using html.escape to ensure that
    user inputs are safely embedded in HTML output.

    Args:
        name (str): The placeholder name to resolve.
        user_values (dict): Mapping from placeholder names to user-provided values.

    Returns:
        str: The HTML-escaped string value for insertion.

    Raises:
        ValueError: If the placeholder name is missing in user_values.
    """
    logger.debug("Resolving value for placeholder: %s", name)
    if name not in user_values:
        logger.error("Missing value for placeholder '%s'", name)
        raise ValueError(f"Missing value for placeholder '{name}'")

    value = user_values[name]
    escaped = html.escape("" if value is None else str(value), quote=True)
    logger.debug("Resolved and escaped value for placeholder '%s'", name)
    return escaped


def create_html_template(html_template: str, user_values: dict) -> str:
    """
    Produce dynamic HTML content by replacing placeholders in a given template
    with user-provided values.

    Placeholders are denoted by double curly braces, e.g., {{ name }}.
    The function scans the template for placeholders, validates each,
    resolves corresponding values from user_values, escapes them for HTML, and
    substitutes them into the template.

    Args:
        html_template (str): An HTML template containing placeholders delimited
            with {{...}}.
        user_values (dict): A mapping of placeholder names to the values to be
            inserted.

    Returns:
        str: The HTML content generated after the placeholder replacement.

    Raises:
        ValueError: If template processing fails or if a placeholder is missing
            in user_values.
    """
    if not isinstance(html_template, str):
        raise ValueError("html_template must be a str")
    if not isinstance(user_values, dict):
        raise ValueError("user_values must be a dict")

    try:
        # Log a preview of detected placeholders to aid debugging.
        detected = [m.group(0) for m in _PLACEHOLDER_PATTERN.finditer(html_template)]
        logger.debug(
            "Starting template processing: %d placeholder(s) detected: %s",
            len(detected),
            detected,
        )

        def _replacer(match: re.Match) -> str:
            name = _extract_placeholder_name(match)
            value = _insert_value(name, user_values)
            logger.debug("Replaced placeholder '%s'", name)
            return value

        result = _PLACEHOLDER_PATTERN.sub(_replacer, html_template)
        logger.debug("Template processing complete")
        return result
    except Exception as exc:
        # Normalize all processing errors to ValueError as specified.
        logger.error("Template processing failed", exc_info=True)
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Template processing failed: {exc}") from exc
