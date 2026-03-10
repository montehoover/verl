import re
from typing import List, Tuple

def find_urls(text: str) -> List[Tuple[str, str]]:
    """
    Scans a string and extracts all URL-like patterns and their schemes.

    Args:
        text: The string to scan for URLs.

    Returns:
        A list of tuples, where each tuple contains the URL and its scheme.
    """
    # Regex to capture the scheme (http, https, ftp) and the rest of the URL.
    # Group 1: scheme (e.g., "http", "https", "ftp")
    # Group 0: full URL
    url_pattern = re.compile(
        r'(http[s]?|ftp)://'  # Scheme (http, https, ftp)
        r'((?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)' # domain name and path
    )
    
    matches = url_pattern.finditer(text)
    results = []
    for match in matches:
        full_url = match.group(0)
        scheme = match.group(1)
        results.append((full_url, scheme))
    return results

if __name__ == '__main__':
    sample_text_with_urls = """
    Visit our website at http://www.example.com for more information.
    You can also check out https://example.org/path?query=param.
    Another link is ftp://files.example.net/data.txt.
    This is not a url: www.missingprotocol.com
    And another one: http://localhost:8000/my/page
    Check this: https://sub.domain.example.co.uk/another/path.html#fragment
    """
    urls_found = find_urls(sample_text_with_urls)
    print("URLs and schemes found:")
    for url, scheme in urls_found:
        print(f"URL: {url}, Scheme: {scheme}")

    sample_text_without_urls = "This is a string with no URLs."
    urls_found_none = find_urls(sample_text_without_urls)
    print("\nURLs and schemes found in text without URLs:")
    print(urls_found_none)

    sample_text_edge_cases = "Text with url http://example.com.And anotherhttps://another.com immediately after. Also ftp://myfiles.org/data"
    urls_found_edge = find_urls(sample_text_edge_cases)
    print("\nURLs and schemes found in edge case text:")
    for url, scheme in urls_found_edge:
        print(f"URL: {url}, Scheme: {scheme}")
