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

# Regex to capture the raw attribute section of a single tag string.
# Captures everything after the tag name up to the closing ">".
_TAG_ATTRS_RE = re.compile(
    r"""
    ^\s*<\s*/?\s*                                      # Opening "<" with optional "/" and whitespace
    [A-Za-z][A-Za-z0-9:_-]*\b                          # Tag name
    (?P<attrs>[\s\S]*?)                                # Non-greedy capture of the rest (attributes area)
    >\s*$                                              # Closing ">"
    """,
    re.DOTALL | re.VERBOSE,
)

# Regex to parse individual attributes within the attributes area.
# Supports:
# - name
# - name="value"
# - name='value'
# - name=value (unquoted)
# - boolean attributes (just "name")
_ATTR_RE = re.compile(
    r"""
    (?:^|(?<=\s))                                      # Start or whitespace boundary
    (?P<name>[A-Za-z_:][A-Za-z0-9:._-]*)               # Attribute name
    (?:\s*=\s*
        (?:
            "(?P<dq>[^"]*)"                            # Double-quoted value
            |
            '(?P<sq>[^']*)'                            # Single-quoted value
            |
            (?P<uq>[^'"`\s<>/=]+)                      # Unquoted value
        )
    )?                                                 # Optional value assignment
    (?=\s|$)                                           # Followed by whitespace or end
    """,
    re.VERBOSE | re.DOTALL,
)

def contains_html(text: str) -> bool:
    """
    Return True if the input string contains suspected HTML-like content,
    such as tags (<tag>, </tag>, <tag/>) or constructs like comments and DOCTYPE.
    """
    if not text:
        return False
    return _HTML_PATTERN.search(text) is not None

def extract_tag_attributes(tag: str) -> dict:
    """
    Extract attributes from a single HTML tag string into a dictionary.
    - Returns {name: value} with:
        - value as the string (for quoted or unquoted values)
        - value as True for boolean attributes (no value specified)
    Example:
        <a href='#'> -> {'href': '#'}
        <input disabled> -> {'disabled': True}
    """
    if not tag or "<" not in tag or ">" not in tag:
        return {}

    m = _TAG_ATTRS_RE.match(tag)
    if not m:
        return {}

    attrs_str = (m.group("attrs") or "").strip()

    # Remove a trailing slash for self-closing tags (e.g., <br /> or <img ... />)
    if attrs_str.endswith("/"):
        attrs_str = attrs_str[:-1].rstrip()

    result = {}
    for am in _ATTR_RE.finditer(attrs_str):
        name = am.group("name")
        if not name:
            continue
        if am.group("dq") is not None:
            value = am.group("dq")
        elif am.group("sq") is not None:
            value = am.group("sq")
        elif am.group("uq") is not None:
            value = am.group("uq")
        else:
            value = True  # Boolean attribute
        result[name] = value

    return result
