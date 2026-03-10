"""
Utilities for detecting HTML tags within text using regular expressions.

This module exposes a single public function, `detect_html_tags`, which
returns all matched HTML tags from the provided input string. The core
regular-expression parsing is implemented as a pure helper function to
improve testability and maintainability.

Design goals:
- Readability: PEP 8–style formatting and strategic blank lines.
- Safety: No exceptions are raised from the public API.
- Flexibility: Supports typical HTML tags, custom elements, and namespace
  prefixes in tag names.
"""

import re

__all__ = ["detect_html_tags"]


# Compiled pattern to match opening, closing, and self-closing HTML tags.
# Notes:
# - Excludes comments (<!-- -->), DOCTYPE (<!DOCTYPE ...>), and processing
#   instructions (<? ... ?>) by requiring the first character after '<' (or
#   '</') to be alphabetic.
# - Supports custom elements (hyphens) and namespaced tags (colon).
_TAG_PATTERN = re.compile(
    r"""
    <\s*/?\s*                          # '<' and optional '/'
    [A-Za-z][A-Za-z0-9:-]*             # tag name
    (?:                                # begin attributes group
        \s+                            # whitespace before attribute
        [A-Za-z_:][A-Za-z0-9:._-]*     # attribute name
        (?:                            # optional attribute value
            \s*=\s*
            (?:
                "[^"]*"                # double-quoted value
              | '[^']*'                # single-quoted value
              | [^'"\s<>]+             # unquoted value (until space or angle)
            )
        )?
    )*                                 # zero or more attributes
    \s*/?\s*>                          # optional '/', then '>'
    """,
    re.VERBOSE,
)


def _parse_html_tags(text: str) -> list[str]:
    """
    Pure function that applies the HTML tag regex to the provided text.

    This function is free of side effects and raises no exceptions.

    Args:
        text (str): Input text to search for HTML tags.

    Returns:
        list[str]: A list of matched HTML tag strings. Returns an empty list
            if no matches or if the input is empty.

    Examples:
        >>> _parse_html_tags("<div class='x'>Hi</div>")
        ["<div class='x'>", "</div>"]
        >>> _parse_html_tags("No tags here")
        []
    """
    return _TAG_PATTERN.findall(text) or []


def detect_html_tags(html_str: str) -> list[str]:
    """
    Identify all HTML tags in a given string using regular expressions.

    This function orchestrates validation and normalization of the input,
    then delegates actual parsing to a pure helper function. It guarantees
    that no exceptions are raised to the caller.

    Args:
        html_str (str): A string representing the input HTML content to be
            analyzed.

    Returns:
        list[str]: A list containing the strings that match as HTML tags
            within the input.

    Examples:
        >>> detect_html_tags("<p id='a'>Text<br/>more</p>")
        ["<p id='a'>", "<br/>", "</p>"]
        >>> detect_html_tags(None)
        []
        >>> detect_html_tags(123)  # coerces to string "123"
        []
    """
    try:
        if html_str is None:
            return []

        if not isinstance(html_str, str):
            try:
                html_str = str(html_str)
            except Exception:
                return []

        return _parse_html_tags(html_str)
    except Exception:
        # Guarantee no exceptions are raised by this function
        return []
