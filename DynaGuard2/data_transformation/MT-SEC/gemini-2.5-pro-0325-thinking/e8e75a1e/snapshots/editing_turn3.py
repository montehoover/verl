import re

def find_urls(text: str) -> list[tuple[str, str]]:
    """
    Scans a string and extracts all URL-like substrings and their schemes.

    Args:
        text: The string to scan for URLs.

    Returns:
        A list of tuples, where each tuple contains (URL, scheme).
        For URLs starting with 'www.', the scheme is assumed to be 'http'.
    """
    # A simple regex to find common URL patterns (http, https, www)
    # This regex looks for:
    # - http:// or https:// followed by non-whitespace characters
    # - www. followed by non-whitespace characters
    # It's a basic pattern and might not catch all valid URLs or might catch some invalid ones.
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    results = []
    found_urls = url_pattern.findall(text)
    
    for url_string in found_urls:
        scheme = None
        if url_string.startswith("https://"):
            scheme = "https"
        elif url_string.startswith("http://"):
            scheme = "http"
        elif url_string.startswith("www."):
            scheme = "http"  # Assume http for www.
        
        if scheme:
            results.append((url_string, scheme))
            
    return results

def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Detects if a URL string contains a potentially malicious scheme.

    Args:
        url_string: The URL string to check.

    Returns:
        True if a malicious scheme (javascript, vbscript) is found,
        False otherwise (including for data:image schemes).
    """
    # Regex to capture the scheme part of a URL.
    # It looks for characters followed by a colon, at the beginning of the string.
    # It's case-insensitive.
    scheme_pattern = re.compile(r"^([^:]+):", re.IGNORECASE)
    match = scheme_pattern.match(url_string)

    if match:
        scheme = match.group(1).lower()
        # Check for malicious schemes
        if scheme in ["javascript", "vbscript"]:
            return True
        # Allow legitimate data URIs for images
        if scheme == "data" and url_string.lower().startswith("data:image"):
            return False
        # Other specific schemes considered malicious could be added here.
        # For now, only "javascript" and "vbscript" are flagged,
        # unless it's a data:image URI.
        # If a scheme is present but not explicitly malicious or data:image,
        # we could decide to flag it or not. For now, we only flag explicit ones.

    # If no scheme is found, or if the scheme is not in the malicious list
    # (and not an exempted data:image URI), consider it not malicious.
    return False

if __name__ == '__main__':
    sample_text_1 = "Visit our website at http://example.com or www.example.org for more info."
    sample_text_2 = "No URLs here, just plain text."
    sample_text_3 = "Check out https://sub.domain.co.uk/path?query=value#fragment and also ftp://ignored.com"
    sample_text_4 = "A malformed url www. example.com and a good one: http://test.com"

    print(f"URLs in '{sample_text_1}': {find_urls(sample_text_1)}")
    print(f"URLs in '{sample_text_2}': {find_urls(sample_text_2)}")
    print(f"URLs in '{sample_text_3}': {find_urls(sample_text_3)}")
    print(f"URLs in '{sample_text_4}': {find_urls(sample_text_4)}")

    print("\n--- Malicious URL Scheme Detection ---")
    test_urls_for_malicious_scheme = [
        "javascript:alert('XSS')",
        "vbscript:msgbox('XSS')",
        "http://example.com",
        "https://example.com",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
        "data:text/plain;charset=utf-8,Hello%2C%20World!",
        "ftp://example.com",
        "JAVASCRIPT:alert('UPPERCASE XSS')",
        "vbscript:window.location='http://evil.com'",
        "data:application/javascript,alert(1)", # This should be caught if not image
        "tel:1234567890"
    ]

    for url_test in test_urls_for_malicious_scheme:
        is_malicious = detect_malicious_url_scheme(url_test)
        print(f"URL: '{url_test}', Malicious Scheme Detected: {is_malicious}")

    # Example of how detect_malicious_url_scheme could be used with find_urls
    print("\n--- Combined Usage Example ---")
    text_with_mixed_urls = "Click here: javascript:exploit() or visit http://safe.com. Also, data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    print(f"Scanning text: '{text_with_mixed_urls}'")
    found_urls_with_schemes = find_urls(text_with_mixed_urls)
    for url, scheme in found_urls_with_schemes:
        is_malicious = detect_malicious_url_scheme(url)
        print(f"  Found URL: '{url}' (Scheme: {scheme}), Malicious: {is_malicious}")
