import re

_COMBINED_PATTERN = re.compile(
    r'(?i)(?:(?P<url>\b(?:https?://|ftp://|www\.)[^\s"\'<>\]\)}]+(?<![.,!?;:]))|'
    r'(?P<email>\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b))'
)

def get_html_tags(text: str):
    """
    Extract all URLs and email addresses from the given text using regular expressions.

    Args:
        text (str): Input string possibly containing URLs and email addresses.

    Returns:
        list[str]: List of URL and email strings found in the text, in scan order.
    """
    if not isinstance(text, str) or not text:
        return []
    return [
        m.group('url') if m.group('url') is not None else m.group('email')
        for m in _COMBINED_PATTERN.finditer(text)
    ]
