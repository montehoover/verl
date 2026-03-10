import re
import logging

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r'^\s*([^(]*)\(([^)]*)\)\s*$')


def _extract_outside_inside(text: str):
    """
    Helper function that performs the regex-based extraction.

    This function attempts to match the input against the pattern:
    'outside(inside)' (allowing leading/trailing whitespace). If matched,
    it returns a tuple of (outside, inside); otherwise, it returns None.

    Notes:
        - Assumes 'text' is a string. Type validation and exception handling
          are delegated to the caller.
        - Emits debug-level logs for inputs, match status, and outputs.

    Args:
        text (str): The input string to be matched.

    Returns:
        tuple[str, str] | None: (outside, inside) if matched; otherwise None.
    """
    logger.debug("_extract_outside_inside input: %r", text)

    match = _PATTERN.match(text)
    if not match:  # guard clause for non-matching input
        logger.debug("_extract_outside_inside no match")
        return None

    result = (match.group(1), match.group(2))
    logger.debug("_extract_outside_inside output: %r", result)
    return result


def match_strings(text: str):
    """
    Match a string of the form 'outside(inside)' and return the captured parts.

    This is the public API that focuses on input/output flow and delegates the
    regex matching to a dedicated helper function for better readability and
    maintainability.

    Behavior:
        - Returns (outside, inside) when the input matches the pattern.
        - Returns None if the input is not a string, does not match,
          or if any unexpected error occurs.
        - Does not raise exceptions.
        - Emits debug-level logs for inputs, outputs, and errors.

    Args:
        text (str): The input string to be matched.

    Returns:
        tuple[str, str] | None: (outside, inside) if matched; otherwise None.
    """
    logger.debug("match_strings input: %r", text)

    # Guard clause for invalid type
    if not isinstance(text, str):
        logger.debug("match_strings early return: input is not a string")
        return None

    try:
        result = _extract_outside_inside(text)
        logger.debug("match_strings output: %r", result)
        return result
    except Exception:
        logger.exception("match_strings encountered an unexpected error")
        return None
