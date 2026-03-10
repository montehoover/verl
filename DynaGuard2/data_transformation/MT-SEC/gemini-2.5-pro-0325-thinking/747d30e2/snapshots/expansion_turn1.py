import re

def find_urls(text: str) -> list[str]:
    """
    Finds all URL-like patterns in a string.

    Args:
        text: The string to search for URLs.

    Returns:
        A list of URL-like patterns found in the text.
    """
    # A common regex for matching URLs. This pattern is not exhaustive but covers many common cases.
    # It looks for http:// or https:// followed by a domain name and path.
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)

if __name__ == '__main__':
    sample_text_with_urls = "Visit our website at http://example.com or check out https://www.another-example.org/path?query=param. Also, ftp://fileserver.com is not matched by this specific regex."
    found_urls = find_urls(sample_text_with_urls)
    print(f"Found URLs: {found_urls}")

    sample_text_without_urls = "This is a string with no URLs."
    found_urls_none = find_urls(sample_text_without_urls)
    print(f"Found URLs (none): {found_urls_none}")

    sample_text_edge_cases = "Text with url.com and www.domain.net but no http/https. Also http://localhost:8000/ and https://sub.domain.co.uk/page.html#anchor"
    found_urls_edge = find_urls(sample_text_edge_cases)
    print(f"Found URLs (edge cases): {found_urls_edge}")
