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

def extract_url_scheme(url: str) -> str:
    """
    Extracts the scheme part of a URL.

    Args:
        url: The URL string.

    Returns:
        The scheme as a string (e.g., 'http', 'https') or 'no_scheme' if not found.
    """
    # Regex to find the scheme at the beginning of the URL, followed by '://'
    # It captures the scheme part (e.g., http, https, ftp, javascript)
    scheme_pattern = re.compile(r'^([a-zA-Z][a-zA-Z0-9+.-]*):(?:/{2,}|javascript:|mailto:)', re.IGNORECASE)
    match = scheme_pattern.match(url)
    if match:
        return match.group(1).lower()  # Return the captured scheme in lowercase
    return 'no_scheme'

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

    print("\n--- Testing extract_url_scheme ---")
    urls_to_test_scheme = [
        "http://example.com",
        "https://www.another-example.org/path?query=param",
        "ftp://fileserver.com/resource",
        "javascript:alert('hello')",
        "mailto:user@example.com",
        "HTTP://CaseSensitive.Com",
        "example.com", # No scheme
        "//cdn.example.com/script.js", # Protocol-relative URL, no scheme by this regex
        "ws://websocket.example.com",
        "tel:+1-555-555-5555"
    ]
    for url_test in urls_to_test_scheme:
        scheme = extract_url_scheme(url_test)
        print(f"URL: '{url_test}', Scheme: '{scheme}'")
