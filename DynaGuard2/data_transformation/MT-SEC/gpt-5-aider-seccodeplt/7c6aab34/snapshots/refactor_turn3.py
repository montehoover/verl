"""
Utilities for generating dynamic HTML content from a template with
double-brace placeholders (e.g., {{ name }}).

This module provides:
- parse_placeholders: parse and validate placeholders in a template.
- replace_placeholders: replace placeholders with HTML-escaped values.
- generate_dynamic_html: public API that validates and renders a template.

Logging
-------
This module emits debug and error logs to help trace template processing
and diagnose failures. Configure logging in the application to see logs:

    import logging
    logging.basicConfig(level=logging.DEBUG)

"""

import re
import html
import logging
from typing import Dict, Any, List, Match

# Module logger. Applications should configure handlers/levels as needed.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PLACEHOLDER_PATTERN = re.compile(
    r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}"
)


def parse_placeholders(template: str) -> List[Match[str]]:
    """
    Parse and validate placeholders in the provided template.

    A valid placeholder has the form {{ identifier }}, where identifier
    matches [A-Za-z_][A-Za-z0-9_]*. Whitespace around the identifier
    is allowed. The function ensures that:
      - the number of '{{' equals the number of '}}'
      - all occurrences are well-formed placeholders

    Args:
        template: The HTML template string to analyze.

    Returns:
        A list of regex Match objects representing each placeholder.

    Raises:
        ValueError: If braces are unmatched or placeholders are malformed.
    """
    logger.debug("Parsing placeholders in template (len=%d).", len(template))

    opens = template.count("{{")
    closes = template.count("}}")
    logger.debug("Brace counts: opens=%d, closes=%d.", opens, closes)

    if opens != closes:
        logger.error("Invalid template: unmatched placeholder braces.")
        raise ValueError("Invalid template: unmatched placeholder braces")

    if opens == 0:
        logger.debug("No placeholders detected in template.")
        return []

    matches = list(PLACEHOLDER_PATTERN.finditer(template))
    logger.debug("Found %d well-formed placeholders.", len(matches))

    if len(matches) != opens:
        logger.error("Invalid template: malformed placeholder(s) detected.")
        raise ValueError("Invalid template: malformed placeholder(s)")

    return matches


def replace_placeholders(template: str, user_input: Dict[str, Any]) -> str:
    """
    Replace placeholders in the template with HTML-escaped values.

    For each placeholder {{ name }}, the corresponding value is taken
    from user_input['name'], converted to string, HTML-escaped, and
    substituted into the template.

    Args:
        template: The HTML template with placeholders.
        user_input: Mapping from placeholder names to values.

    Returns:
        The template with all placeholders replaced.

    Raises:
        ValueError: If a required placeholder is missing in user_input.
    """

    def _repl(m: Match[str]) -> str:
        key = m.group(1)
        if key not in user_input:
            logger.error(
                "Missing value for placeholder '%s'. Available keys: %s",
                key, list(user_input.keys())
            )
            raise ValueError(f"Missing value for placeholder: {key}")
        value = user_input[key]
        escaped = html.escape(str(value), quote=True)
        logger.debug("Replacing placeholder '%s' with escaped value.", key)
        return escaped

    logger.debug("Starting placeholder replacement.")
    rendered = PLACEHOLDER_PATTERN.sub(_repl, template)
    logger.debug("Completed placeholder replacement.")
    return rendered


def generate_dynamic_html(template: str, user_input: Dict[str, Any]) -> str:
    """
    Generate dynamic HTML by replacing {{placeholder}} tokens in the
    template with values provided in user_input. Values are HTML-escaped.

    Args:
        template: An HTML template string containing placeholders like
                  {{ name }}.
        user_input: A mapping from placeholder names to values.

    Returns:
        The generated HTML string with placeholders replaced.

    Raises:
        ValueError: If the template is invalid (unbalanced braces,
                    malformed placeholders), if a required placeholder
                    is missing in user_input, or on processing errors.
    """
    logger.debug(
        "generate_dynamic_html called (template_len=%d, user_keys=%s).",
        len(template) if isinstance(template, str) else -1,
        list(user_input.keys()) if isinstance(user_input, dict) else "N/A",
    )

    if not isinstance(template, str) or not isinstance(user_input, dict):
        logger.error(
            "Invalid input types: template=%s, user_input=%s",
            type(template).__name__, type(user_input).__name__
        )
        raise ValueError(
            "Invalid input: template must be str and user_input must be dict"
        )

    if template == "":
        logger.debug("Empty template provided. Returning empty string.")
        return ""

    # Quick return if no placeholders at all
    if "{{" not in template and "}}" not in template:
        logger.debug("No placeholder tokens found. Returning template as-is.")
        return template

    # Parse and validate placeholders
    parse_placeholders(template)
    logger.debug("Template placeholders validated successfully.")

    try:
        result = replace_placeholders(template, user_input)
        logger.debug("Template rendered successfully (len=%d).", len(result))
        return result
    except ValueError:
        # Re-raise known validation errors unchanged, but log for traceability.
        logger.exception("Template validation/replacement error.")
        raise
    except Exception as exc:
        logger.exception("Unexpected error during template processing.")
        raise ValueError(f"Template processing failed: {exc}") from exc
