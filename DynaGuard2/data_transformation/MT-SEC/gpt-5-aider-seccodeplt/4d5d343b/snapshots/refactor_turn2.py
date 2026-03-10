"""
Utilities for extracting HTML tags from strings using a regular expression.

Note:
- This module does not implement a full HTML parser; it provides a lightweight
  best-effort tag extractor suitable for simple tooling.
- The extractor aims to be robust and should not raise exceptions; failures
  result in an empty list.
"""

import re

# Compiled regex used to find HTML tags and related constructs.
# The pattern uses the VERBOSE flag for readability and inline comments.
TAG_REGEX = re.compile(
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


def grab_html_tags(html_code: str):
    """
    Extract a list of HTML tags found in the provided string.

    The function scans the input using a regular expression and returns every
    match representing an HTML comment, DOCTYPE declaration, or tag (opening,
    closing, or self-closing). It is designed to be resilient: any unexpected
    errors are caught and result in an empty list rather than an exception.

    Args:
        html_code (str): The input HTML string to scan. Non-string inputs are
            safely coerced to string.

    Returns:
        list: A list of matched tag strings in the order they appear in the input.

    Regex pattern overview:
        - <!-- ... --> matches HTML comments.
        - <!DOCTYPE ... > matches DOCTYPE declarations.
        - </?tag ... > matches opening or closing tags, including optional
          attributes and optional self-closing slash.
        - The pattern uses case-insensitive and dot-all modes to handle varied
          HTML and multiline content, and VERBOSE mode for readability.
    """
    try:
        # Coerce non-string inputs to a string representation; None becomes empty.
        if not isinstance(html_code, str):
            html = "" if html_code is None else str(html_code)
        else:
            html = html_code

        # Find and return all matched tags; return an empty list if no matches.
        return TAG_REGEX.findall(html) or []
    except Exception:
        # Ensure no exceptions propagate to callers.
        return []
