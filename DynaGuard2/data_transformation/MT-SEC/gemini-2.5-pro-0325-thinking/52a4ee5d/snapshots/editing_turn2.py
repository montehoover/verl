import re

def find_urls(text: str) -> list[tuple[str, str]]:
    """
    Scans a string and extracts all URL-like patterns and their schemes.

    Args:
        text: The string to scan.

    Returns:
        A list of tuples, where each tuple contains the URL and its scheme.
    """
    # A common regex for matching URLs. This pattern captures the scheme (http, https, ftp)
    # and the rest of the URL. It's not exhaustive and might not
    # capture all possible URL formats or might capture some non-URLs.
    # It looks for scheme, optional www, domain name, and path/query.
    url_pattern = re.compile(
        r'(http[s]?|ftp)://((?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
    )
    matches = url_pattern.finditer(text)
    results = []
    for match in matches:
        scheme = match.group(1)
        full_url = match.group(0) # group(0) is the entire matched string
        results.append((full_url, scheme))
    return results

if __name__ == '__main__':
    sample_text_with_urls = """
    Visit our site at http://www.example.com.
    You can also check https://example.org/path?query=value.
    Another one is ftp://files.example.net/data.
    This is not a url: www.justsometext.
    But this is: http://localhost:8000
    And this: https://sub.domain.co.uk/page.html#anchor
    Invalid: http//missing-colon.com
    Edge case: http://example.com/!@#$%^&*()_+
    """
    urls_found = find_urls(sample_text_with_urls)
    print("URLs and schemes found:")
    for url, scheme in urls_found:
        print(f"URL: {url}, Scheme: {scheme}")

    sample_text_no_urls = "This is a string with no URLs."
    urls_not_found = find_urls(sample_text_no_urls)
    print("\nURLs and schemes found in text with no URLs:")
    print(urls_not_found)

    sample_text_edge_cases = "Check http://127.0.0.1 or https://[::1]/ (ipv6 not handled by this regex well)"
    urls_edge_cases = find_urls(sample_text_edge_cases)
    print("\nURLs and schemes found in edge case text:")
    for url, scheme in urls_edge_cases:
        print(f"URL: {url}, Scheme: {scheme}")
