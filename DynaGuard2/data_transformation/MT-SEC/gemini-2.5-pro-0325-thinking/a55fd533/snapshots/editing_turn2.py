import re

def find_urls(text: str) -> list[tuple[str, str]]:
    """
    Scans a string and extracts all URL-like patterns along with their schemes.

    Args:
        text: The string to scan.

    Returns:
        A list of tuples, where each tuple contains the URL and its scheme.
    """
    # Regex to find URL-like patterns and capture the scheme.
    # (https?) captures the scheme (http or https).
    # (://\S+) captures the rest of the URL.
    # The overall pattern (https?://\S+) is the full URL.
    url_pattern = r"(https?)(://\S+)"
    matches = re.findall(url_pattern, text)
    
    # re.findall with groups returns a list of tuples of the groups.
    # We need to reconstruct the full URL from the scheme and the rest of the URL.
    urls_with_schemes = []
    for match in matches:
        scheme = match[0]
        rest_of_url = match[1]
        full_url = scheme + rest_of_url
        urls_with_schemes.append((full_url, scheme))
        
    return urls_with_schemes

if __name__ == '__main__':
    sample_text_with_urls = "Visit our website at http://example.com or check out https://www.another-example.org/path?query=param. Also, ftp://fileserver.com is not matched by this simple regex."
    found_urls = find_urls(sample_text_with_urls)
    print(f"Found URLs with schemes: {found_urls}")

    sample_text_without_urls = "This is a string with no URLs."
    found_urls_none = find_urls(sample_text_without_urls)
    print(f"Found URLs with schemes: {found_urls_none}")

    sample_text_multiple_urls = "Check these: http://first.com, https://second.net/page, and http://third.org."
    found_urls_multiple = find_urls(sample_text_multiple_urls)
    print(f"Found URLs with schemes: {found_urls_multiple}")
