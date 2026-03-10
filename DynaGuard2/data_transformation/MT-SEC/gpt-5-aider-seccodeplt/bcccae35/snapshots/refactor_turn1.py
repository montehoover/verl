import re
from typing import List


# Precompiled regular expression to match HTML tags, including:
# - Standard tags: <div>, <a href="...">, </p>, <br/>, etc.
# - Declarations: <!DOCTYPE html>, other <! ... >
# - CDATA sections: <![CDATA[ ... ]]>
# - Comments: <!-- ... -->
_TAG_RE = re.compile(
    r"<!--[\s\S]*?-->|"  # HTML comments
    r"<!\[CDATA\[[\s\S]*?\]\]>|"  # CDATA sections
    r"<![^>]*>|"  # Declarations (e.g., DOCTYPE)
    r"</?\s*[A-Za-z][A-Za-z0-9:\-]*"  # Tag name (opening or closing)
    r"(?:\s+(?:[A-Za-z_:][A-Za-z0-9_\-.:]*)"  # Attribute name
    r"(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^'\"\s>/=]+))?)*"  # Attribute value
    r"\s*/?>",
    re.DOTALL,
)


def parse_html_tags(html: str) -> List[str]:
    """
    Extract all HTML tags from the given string using a regular expression.

    Args:
        html: The input HTML string to be parsed.

    Returns:
        A list of strings, each representing an HTML tag found in the input.
        Never raises exceptions; returns an empty list on error.
    """
    try:
        if not isinstance(html, str):
            html = "" if html is None else str(html)
        return _TAG_RE.findall(html)
    except Exception:
        return []
