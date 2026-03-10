import re
from typing import List, Tuple

def find_urls(text: str) -> List[Tuple[str, str]]:
    """
    Scans a string and extracts all the URL-like patterns it contains,
    along with their schemes.

    Args:
        text: The string to scan.

    Returns:
        A list of tuples, where each tuple contains the URL and its scheme.
    """
    # Regex to capture the scheme (http, https, ftp) and the rest of the URL.
    # Group 1: scheme (e.g., "http", "https", "ftp")
    # Group 0: full URL
    url_pattern = re.compile(
        r'(http[s]?|ftp)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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
    Visit our website at https://www.example.com for more information.
    You can also check out http://example.org/about or ftp://files.example.net/data.
    Some invalid urls like http://localhost or just www.google.com might not be caught perfectly
    or might be caught depending on the regex strictness.
    Another one: https://sub.domain.example.co.uk/path?query=value#fragment
    """
    urls_found = find_urls(sample_text_with_urls)
    print("URLs and schemes found:")
    for url, scheme in urls_found:
        print(f"URL: {url}, Scheme: {scheme}")

    sample_text_without_urls = "This is a string with no URLs."
    urls_not_found = find_urls(sample_text_without_urls)
    print("\nURLs and schemes found in text without URLs:")
    if not urls_not_found:
        print("(No URLs found)")
    else:
        for url, scheme in urls_not_found:
            print(f"URL: {url}, Scheme: {scheme}")
