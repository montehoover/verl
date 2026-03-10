"""
Utilities for extracting HTML tags from text.

This module provides:
- get_tag_regex: Returns a compiled regular expression used to extract HTML
  tags (including declarations, comments, and CDATA sections).
- parse_html_tags: Uses the compiled regex to extract tags from an input HTML
  string. The function includes logging to help debug input and output.

Note: This module does not configure logging handlers. To see logs, configure
the logging system in the application consuming this library.
"""

import logging
import re
from functools import lru_cache
from typing import List, Pattern

# Module-level logger for this module.
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_tag_regex() -> Pattern[str]:
    """
    Build and return a compiled regular expression that matches HTML tags.

    The pattern matches:
      - HTML comments:           <!-- ... -->
      - CDATA sections:          <![CDATA[ ... ]]>
      - Declarations:            <!DOCTYPE html>, other <! ... >
      - Start/end/self-closing tags with attributes:
          <div>, </p>, <br/>, <a href="...">, <img src='...' />, etc.

    Implementation notes:
      - re.DOTALL is used so '.' matches newlines, enabling multi-line tags.
      - re.VERBOSE is used to allow inline comments and spacing for readability.
      - Attribute values can be double-quoted, single-quoted, or unquoted
        (stopping at whitespace or specific delimiter characters).
    """
    pattern = r"""
        <!--[\s\S]*?-->                         # HTML comments
        | <!\[CDATA\[[\s\S]*?\]\]>              # CDATA sections
        | <![^>]*>                              # Declarations (e.g., DOCTYPE)
        | </?\s*[A-Za-z][A-Za-z0-9:\-]*         # Tag name (open/close)
            (?:                                 # Zero or more attributes
                \s+
                (?:[A-Za-z_:][A-Za-z0-9_\-.:]*)# Attribute name
                (?:\s*=\s*
                    (?: "[^"]*" | '[^']*' | [^'"\s>/=]+ )
                )?
            )*
            \s*/?>                              # Optional self-closing '/',
                                                # then '>'
    """
    return re.compile(pattern, re.DOTALL | re.VERBOSE)


def parse_html_tags(html: str) -> List[str]:
    """
    Extract all HTML tags from the given string using a regular expression.

    This function matches:
      - Standard start/end tags (e.g., <div>, </p>, <br/>, <a href="...">)
      - Declarations (e.g., <!DOCTYPE html>, other <! ... > constructs)
      - CDATA sections (e.g., <![CDATA[ ... ]]>)
      - HTML comments (e.g., <!-- ... -->)

    Logging:
      - Logs the input HTML string at DEBUG level.
      - Logs the list of extracted tags at DEBUG level.
      - Logs exceptions at ERROR level with stack trace.

    The function is defensive and will not raise exceptions; on any unexpected
    error, it returns an empty list.

    Args:
        html: The input HTML string to be parsed.

    Returns:
        A list of strings, each representing a complete HTML tag found in the
        input string.
    """
    try:
        logger.debug("parse_html_tags: input=%r", html)

        if not isinstance(html, str):
            html = "" if html is None else str(html)

        regex = get_tag_regex()
        tags = regex.findall(html)

        logger.debug(
            "parse_html_tags: extracted %d tag(s): %r",
            len(tags),
            tags,
        )
        return tags
    except Exception:
        logger.exception("parse_html_tags: error while parsing HTML.")
        return []
