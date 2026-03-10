import re

def detect_html_tags(content: str) -> list[str]:
    """
    Captures all URLs and email addresses in a given string using regular expressions.

    Args:
        content: The input text to analyze.

    Returns:
        A list of strings where each string is a detected URL or email address.
    """
    # A common regex pattern to match URLs.
    # This pattern looks for http/https followed by allowed URL characters.
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\'(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, content)

    # A common regex pattern to match email addresses.
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, content)

    return urls + emails
