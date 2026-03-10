import re

# Compiled pattern to match opening, closing, and self-closing HTML tags.
# - Excludes comments <!-- -->, DOCTYPE <!DOCTYPE ...>, and processing instructions <? ... ?>
# - Supports custom elements (hyphens) and namespaced tags (colon)
_TAG_PATTERN = re.compile(
    r'<\s*/?\s*'                              # '<' and optional '/'
    r'[A-Za-z][A-Za-z0-9:-]*'                 # tag name
    r'(?:'                                    # begin attributes group
    r'(?:\s+[A-Za-z_:][A-Za-z0-9:._-]*'       # attribute name
    r'(?:\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\'"\s<>]+))?'  # optional value
    r')'                                       # end one attribute
    r')*'                                      # zero or more attributes
    r'\s*/?\s*>'                               # optional '/', '>'
)

def _parse_html_tags(text: str) -> list[str]:
    """
    Pure function that applies the HTML tag regex to the provided text.

    Args:
        text (str): Input text to search.

    Returns:
        list[str]: A list of matched HTML tag strings. Returns an empty list if no matches.
    """
    return _TAG_PATTERN.findall(text) or []


def detect_html_tags(html_str: str) -> list[str]:
    """
    Identify all HTML tags in a given string using regular expressions.

    Args:
        html_str (str): A string representing the input HTML content to be analyzed.

    Returns:
        list[str]: A list containing the strings that match as HTML tags within the input.
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
