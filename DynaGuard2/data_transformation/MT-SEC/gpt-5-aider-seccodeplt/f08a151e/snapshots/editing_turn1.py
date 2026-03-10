import re

_URL_PATTERN = re.compile(
    r'(?i)\b(?:https?://|ftp://|www\.)[^\s"\'<>\]\)}]+(?<![.,!?;:])'
)

def get_html_tags(text: str):
    """
    Extract all URLs from the given text using regular expressions.

    Args:
        text (str): Input string possibly containing URLs.

    Returns:
        list[str]: List of URL strings found in the text, in scan order.
    """
    if not isinstance(text, str) or not text:
        return []
    return _URL_PATTERN.findall(text)
