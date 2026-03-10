import re

def get_html_tags(text: str) -> list[str]:
    """
    Extracts all URLs and email addresses found in a given string of text using regular expressions.

    Args:
        text: The input string.

    Returns:
        A list of URLs and email addresses identified in the text.
    """
    # Regex for matching URLs
    url_pattern = re.compile(
        r'https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)', re.IGNORECASE)
    urls = re.findall(url_pattern, text)

    # Regex for matching email addresses
    # This is a common, relatively simple regex for email validation.
    email_pattern = re.compile(
        r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9]"
        r"(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*", re.IGNORECASE)
    emails = re.findall(email_pattern, text)

    return urls + emails
