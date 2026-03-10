"""
Utilities for extracting HTML tags from text.

This module provides a single function, parse_html_tags, which uses a compiled
regular expression to extract all HTML tags (including declarations, comments,
and CDATA sections) from an input string.
"""

import re
from typing import List

# Compiled regular expression that matches the following constructs:
# - HTML comments:           <!-- ... -->
# - CDATA sections:          <![CDATA[ ... ]]>
# - Declarations:            <!DOCTYPE html>, other <! ... >
# - Start/end/self-closing tags with attributes:
#     <div>, </p>, <br/>, <a href="...">, <img src='...' />, etc.
#
# Notes on the pattern:
# - We use re.DOTALL so that '.' matches newlines, allowing multi-line tags.
# - We use re.VERBOSE to add inline comments and spacing for readability.
# - Attribute values can be double-quoted, single-quoted, or unquoted
#   (stopping at whitespace or specific delimiter characters).
_TAG_RE = re.compile(
    r"""
    <!--[\s\S]*?-->                             # HTML comments
    | <!\[CDATA\[[\s\S]*?\]\]>                  # CDATA sections
    | <![^>]*>                                  # Declarations (e.g., DOCTYPE)
    | </?\s*[A-Za-z][A-Za-z0-9:\-]*             # Tag name (open/close)
        (?:                                     # Zero or more attributes
            \s+
            (?:[A-Za-z_:][A-Za-z0-9_\-.:]*)    # Attribute name
            (?:\s*=\s*
                (?: "[^"]*" | '[^']*' | [^'"\s>/=]+ )
            )?
        )*
        \s*/?>                                  # Optional self-closing '/', then '>'
    """,
    re.DOTALL | re.VERBOSE,
)


def parse_html_tags(html: str) -> List[str]:
    """
    Extract all HTML tags from the given string using a regular expression.

    This function matches:
      - Standard start/end tags (e.g., <div>, </p>, <br/>, <a href="...">)
      - Declarations (e.g., <!DOCTYPE html>, other <! ... > constructs)
      - CDATA sections (e.g., <![CDATA[ ... ]]>)
      - HTML comments (e.g., <!-- ... -->)

    The function is defensive and will not raise exceptions; on any unexpected
    error, it returns an empty list.

    Args:
        html: The input HTML string to be parsed.

    Returns:
        A list of strings, each representing a complete HTML tag found in the
        input string.
    """
    try:
        if not isinstance(html, str):
            html = "" if html is None else str(html)
        return _TAG_RE.findall(html)
    except Exception:
        return []
