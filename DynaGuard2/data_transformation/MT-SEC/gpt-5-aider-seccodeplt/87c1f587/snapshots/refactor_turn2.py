"""
HTML tag finder utility.

This module provides a function, find_html_tags, that locates HTML tags within
a string using a carefully constructed regular expression. It is a pragmatic,
regex-based helper and not a full HTML parser.
"""

import re


# Precompiled regular expression to match HTML element tags:
# - Start tags, e.g., <div>, <a href="...">, <img src=x alt='y' />
# - End tags, e.g., </div>
# - Self-closing tags, e.g., <br/>, <img />
#
# Pattern highlights:
# - Tag name: starts with a letter, followed by letters, digits, or these
#   characters: ':', '_', '-'. Namespaces like svg:rect are supported.
# - Optional leading slash '/' allows matching end tags.
# - Attributes: zero or more name[=value] pairs with flexible whitespace.
#   Values can be double-quoted, single-quoted, or unquoted (limited set).
# - Optional trailing slash before '>' allows self-closing tags.
# - Uses re.VERBOSE for readability and re.IGNORECASE for case-insensitivity.
_TAG_RE = re.compile(
    r"""
    <
        (?:                                     # Entire tag (non-capturing)
            /?[A-Za-z][A-Za-z0-9:_-]*           # Optional '/' + tag name
            (?:\s+                              # One or more spaces before attr
                [A-Za-z_:][A-Za-z0-9:._-]*      # Attribute name
                (?:\s*=\s*                      # Optional '=' with spaces
                    (?:
                        "[^"]*"                 # Double-quoted value
                      | '[^']*'                 # Single-quoted value
                      | [^\s"'=<>`]+            # Unquoted value
                    )
                )?
            )*                                  # Zero or more attributes
            \s*/?                               # Optional trailing slash
        )
    >
    """,
    re.VERBOSE | re.IGNORECASE,
)


def find_html_tags(html_content: str) -> list:
    """
    Identify all HTML tags in the given string using regular expressions.

    This function matches start tags, end tags, and self-closing tags. It does
    not attempt full HTML validation or parsing; it simply extracts substrings
    that look like HTML tags according to the regex.

    Args:
        html_content: The input HTML content to analyze.

    Returns:
        A list of strings, each being a matched HTML tag. The function is
        defensive and will not raise exceptions; on error, it returns an
        empty list.

    Examples:
        >>> find_html_tags('<div class="x">Hello</div>')
        ['<div class="x">', '</div>']

        >>> find_html_tags('No tags here')
        []
    """
    try:
        # Coerce non-string inputs to string safely. Treat None as empty.
        text = (
            html_content
            if isinstance(html_content, str)
            else ("" if html_content is None else str(html_content))
        )

        # Return all matched tags. If no matches, findall returns an empty list.
        return _TAG_RE.findall(text)
    except Exception:
        # Ensure no exceptions escape; return an empty list on any error.
        return []
