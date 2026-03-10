import re

# Precompiled regex to detect HTML-like patterns:
# - Comments: <!-- ... -->
# - Doctype: <!DOCTYPE ...>
# - Processing instructions: <? ... ?>
# - Tags: <tag ...>, </tag>, <tag/>, with optional attributes
_HTML_PATTERN = re.compile(
    r"""
    (?:<!--[\s\S]*?-->)                                # HTML comment
    |
    (?:<!DOCTYPE\b[^>]*>)                              # DOCTYPE
    |
    (?:<\?[\s\S]*?\?>)                                 # Processing instruction
    |
    (?:</?\s*[A-Za-z][A-Za-z0-9:_-]*                   # Tag name
        (?:\s+
            [A-Za-z_:][A-Za-z0-9:._-]*                 # Attribute name
            (?:\s*=\s*
                (?:
                    "[^"]*"                            # Double-quoted value
                    |
                    '[^']*'                            # Single-quoted value
                    |
                    [^'"\s<>/=]+                       # Unquoted value
                )
            )?
        )*
        \s*/?\s*>
    )
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

def contains_html(text: str) -> bool:
    """
    Return True if the input string contains suspected HTML-like content,
    such as tags (<tag>, </tag>, <tag/>) or constructs like comments and DOCTYPE.
    """
    if not text:
        return False
    return _HTML_PATTERN.search(text) is not None
