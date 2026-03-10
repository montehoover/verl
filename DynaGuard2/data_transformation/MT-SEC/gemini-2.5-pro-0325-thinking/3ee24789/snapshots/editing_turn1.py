import re

def detect_html_tags(content: str) -> list[str]:
    """
    Captures all URLs in a given string using regular expressions.

    Args:
        content: The input text to analyze.

    Returns:
        A list of strings where each string is a detected URL.
    """
    # A common regex pattern to match URLs.
    # This pattern looks for http/https followed by allowed URL characters.
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\'(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, content)
    return urls
