import re

# Precompiled regex to match HTML element tags (start, end, and self-closing)
_TAG_RE = re.compile(
    r"""
    <
        (?:                                     # start of tag
            /?[A-Za-z][A-Za-z0-9:_-]*           # optional slash + tag name
            (?:\s+                              # attributes section
                [A-Za-z_:][A-Za-z0-9:._-]*      # attribute name
                (?:\s*=\s*
                    (?:
                        "[^"]*"                 # double-quoted value
                      | '[^']*'                 # single-quoted value
                      | [^\s"'=<>`]+            # unquoted value
                    )
                )?
            )*                                  # zero or more attributes
            \s*/?                               # optional trailing slash (self-closing)
        )
    >
    """,
    re.VERBOSE | re.IGNORECASE,
)

def find_html_tags(html_content: str) -> list:
    """
    Identify all HTML tags in the given string using regular expressions.

    Args:
        html_content: A string representing the input HTML content to be analyzed.

    Returns:
        A list of strings that are matched HTML tags within the input. Never raises.
    """
    try:
        s = html_content if isinstance(html_content, str) else ("" if html_content is None else str(html_content))
        return _TAG_RE.findall(s) or []
    except Exception:
        # Ensure no exceptions escape; return an empty list on any error.
        return []
