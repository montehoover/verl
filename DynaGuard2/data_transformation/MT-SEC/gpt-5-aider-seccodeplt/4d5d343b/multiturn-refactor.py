"""
Utilities for extracting HTML tags from strings using a regular expression.

Note:
- This module does not implement a full HTML parser; it provides a lightweight
  best-effort tag extractor suitable for simple tooling.
- The extractor aims to be robust and should not raise exceptions; failures
  result in an empty list.
- Debug logging is provided to record the input and extracted tags.
"""

import logging
import re
from functools import lru_cache
from typing import List, Pattern

# Module-level logger for this utility.
LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_tag_regex() -> Pattern:
    """
    Build and return the compiled regular expression used to find HTML tags.

    The regex is compiled with VERBOSE for readability and includes the following
    components:
        - <!-- ... -->: Matches HTML comments.
        - <!DOCTYPE ... >: Matches DOCTYPE declarations.
        - </?tag ... >: Matches opening or closing tags, including optional
          attributes and an optional self-closing slash.

    Flags:
        - re.IGNORECASE: Case-insensitive matching for tag names and constructs.
        - re.DOTALL: The '.' metacharacter matches newline characters.
        - re.VERBOSE: Allows whitespace and comments inside the pattern.

    Returns:
        Pattern: A compiled regex pattern object for tag extraction.
    """
    return re.compile(
        r"""
        <!--.*?-->                              # HTML comments
        |                                       # or
        <!DOCTYPE[^>]*>                         # DOCTYPE declarations
        |                                       # or
        </?[A-Za-z][A-Za-z0-9:-]*               # Opening/closing tag and name
            (?:\s+[^<>]*?)?                     # Optional attributes (non-greedy)
            \s*/?>                              # Optional whitespace and '>' or '/>'
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )


def grab_html_tags(html_code: str) -> List[str]:
    """
    Extract a list of HTML tags found in the provided string.

    The function scans the input using a regular expression and returns every
    match representing an HTML comment, DOCTYPE declaration, or tag (opening,
    closing, or self-closing). It is designed to be resilient: any unexpected
    errors are caught and result in an empty list rather than an exception.
    Debug-level logs record the input and the extracted tag list.

    Args:
        html_code (str): The input HTML string to scan. Non-string inputs are
            safely coerced to string.

    Returns:
        list[str]: A list of matched tag strings in the order they appear in the
        input.
    """
    try:
        # Coerce non-string inputs to a string representation; None becomes empty.
        if not isinstance(html_code, str):
            html = "" if html_code is None else str(html_code)
        else:
            html = html_code

        # Log the input HTML string for debugging purposes.
        LOGGER.debug("grab_html_tags input: %s", html)

        # Find and return all matched tags; log the extracted tags.
        pattern = get_tag_regex()
        tags = pattern.findall(html) or []

        LOGGER.debug("grab_html_tags extracted %d tag(s): %s", len(tags), tags)

        return tags
    except Exception as exc:  # noqa: BLE001 - Explicitly broad to ensure robustness.
        # Ensure no exceptions propagate to callers and record the error.
        LOGGER.exception("grab_html_tags encountered an error: %s", exc)
        return []
