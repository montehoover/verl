import re

def get_html_tags(text: str) -> list[str]:
    """
    Extracts all URLs found in a given string of text using regular expressions.

    Args:
        text: The input string.

    Returns:
        A list of URLs identified in the text.
    """
    # A common regex for matching URLs. This can be adjusted for more specific needs.
    # This regex looks for http/https, optional www, domain name, and path/query parameters.
    url_pattern = re.compile(
        r'https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)', re.IGNORECASE)
    return re.findall(url_pattern, text)
