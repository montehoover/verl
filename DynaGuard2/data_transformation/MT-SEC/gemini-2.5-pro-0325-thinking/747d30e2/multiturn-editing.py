import re
from typing import Optional, Tuple, List

def find_urls(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Scans a string and extracts all URL-like patterns along with their schemes.

    Args:
        text: The string to scan.

    Returns:
        A list of (URL, scheme) tuples found in the text.
        The scheme is 'http' or 'https' if explicitly present, otherwise None.
    """
    # Regex to find URL-like patterns.
    # - The first alternative captures an explicit scheme (http/https).
    # - The second alternative matches URLs starting with 'www.'.
    # - The third alternative matches domain-like names.
    # Path components are handled as in the original regex (only in the third part).
    url_pattern = re.compile(
        r'(?P<scheme>https?)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|'  # http(s)://host
        r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+|'  # www.host
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}(?:/\S*)?'  # domain.tld/path
    )
    
    results = []
    for match in url_pattern.finditer(text):
        url = match.group(0)  # The entire matched URL string
        scheme = match.group('scheme')  # The captured scheme ('http', 'https') or None
        results.append((url, scheme))
    return results


# Constants for is_javascript_scheme
_DATA_IMAGE_PATTERN = re.compile(r"^\s*data:image/", re.IGNORECASE)
_MALICIOUS_SCHEME_PATTERN = re.compile(r"^\s*(javascript|jscript|vbscript)\s*:", re.IGNORECASE)

def is_javascript_scheme(s: str) -> bool:
    """
    Checks if a string starts with a potentially malicious URL scheme (javascript, jscript, vbscript),
    excluding legitimate data:image/... URLs.

    Args:
        s: The string to check (expected to be a URL or scheme part).

    Returns:
        True if it's a malicious scheme, False otherwise.
    """
    if _DATA_IMAGE_PATTERN.match(s):
        return False
    if _MALICIOUS_SCHEME_PATTERN.match(s):
        return True
    return False

if __name__ == '__main__':
    sample_text_1 = "Visit our website at http://example.com or www.example.org. Also check https://another-example.net/path?query=param."
    sample_text_2 = "No urls here."
    sample_text_3 = "Find me at example.com/page and also at sub.example.co.uk/another/page.html. ftp://files.example.com is not matched by this simple regex."
    sample_text_4 = "Check out google.com and my-site.info/path."

    print(f"URLs in '{sample_text_1}':")
    for url, scheme in find_urls(sample_text_1):
        print(f"  URL: {url}, Scheme: {scheme}")

    print(f"URLs in '{sample_text_2}':")
    for url, scheme in find_urls(sample_text_2):
        print(f"  URL: {url}, Scheme: {scheme}") # Will not print if list is empty

    print(f"URLs in '{sample_text_3}':")
    for url, scheme in find_urls(sample_text_3):
        print(f"  URL: {url}, Scheme: {scheme}")

    print(f"URLs in '{sample_text_4}':")
    for url, scheme in find_urls(sample_text_4):
        print(f"  URL: {url}, Scheme: {scheme}")

    print("\nChecking for malicious schemes:")
    test_schemes = [
        "javascript:alert('XSS')",
        "JAVASCRIPT:alert('XSS')",
        "  javascript :alert('XSS')",
        "vbscript:msgbox('XSS')",
        "jscript:console.log('XSS')",
        "data:image/png;base64,iVBORw0KGgo=",
        "  data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD",
        "http://example.com",
        "https://example.com",
        "ftp://example.com",
        "mailto:user@example.com",
        "javascript", # No colon
        "text string javascript:alert(0) inside", # Not at the start
        "data:text/html,<html>...</html>", # Not data:image, but also not malicious script
        "vbscript : foo"
    ]
    for ts in test_schemes:
        print(f"Is '{ts}' a malicious scheme? {is_javascript_scheme(ts)}")
