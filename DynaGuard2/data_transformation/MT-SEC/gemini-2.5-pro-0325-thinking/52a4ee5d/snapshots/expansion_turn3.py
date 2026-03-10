import re

def extract_url_candidates(text: str) -> list[str]:
    """
    Finds and returns a list of URL-like patterns within the given text.

    Args:
        text: The input string to search for URLs.

    Returns:
        A list of strings, where each string is a potential URL.
    """
    # A common, but not exhaustive, regex for URLs.
    # This regex looks for http/https, optional www, domain name, and path/query parameters.
    url_pattern = re.compile(
        r'http[s]?://'  # http:// or https://
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # domain name and path
    )
    return url_pattern.findall(text)

def identify_url_scheme(url_candidate: str) -> str:
    """
    Identifies the scheme of a given URL candidate.

    Args:
        url_candidate: The URL string to analyze.

    Returns:
        The scheme (e.g., 'http', 'https', 'ftp') or 'unknown' if no scheme is found.
    """
    # Regex to find a scheme at the beginning of the string, followed by '://'
    # Schemes consist of letters, digits, plus (+), period (.), or hyphen (-)
    scheme_pattern = re.compile(r'^([a-zA-Z][a-zA-Z0-9+.-]*):')
    match = scheme_pattern.match(url_candidate)
    if match:
        return match.group(1).lower()  # Return the scheme in lowercase
    return 'unknown'

def detect_js_scheme(url_string: str) -> bool:
    """
    Detects potentially harmful URL schemes like 'javascript:' or 'vbscript:',
    while ignoring valid image data URLs.

    Args:
        url_string: The URL string to examine.

    Returns:
        True if a harmful scheme is present, False otherwise.
    """
    # Normalize to lowercase and strip whitespace for case-insensitive and robust checks
    normalized_url = url_string.strip().lower()

    # Ignore valid data:image/... URLs specifically
    if normalized_url.startswith('data:image/'):
        return False

    # Check for harmful schemes like javascript: or vbscript:
    # The regex looks for 'javascript' or 'vbscript' followed by a colon,
    # at the beginning of the string.
    harmful_scheme_pattern = re.compile(r'^(javascript|vbscript):')
    if harmful_scheme_pattern.match(normalized_url):
        return True

    return False

if __name__ == '__main__':
    sample_text_with_urls = """
    Visit our website at http://www.example.com for more information.
    You can also check out https://example.org/path?query=param.
    Another link is http://sub.example.co.uk/another/path.
    Not a url: example.com. And this ftp://old.server.com is not matched by this regex.
    """
    candidates = extract_url_candidates(sample_text_with_urls)
    print("Found URL candidates:")
    for url in candidates:
        print(url)

    sample_text_without_urls = "This is a string with no URLs."
    candidates_none = extract_url_candidates(sample_text_without_urls)
    print(f"\nFound URL candidates in '{sample_text_without_urls}': {candidates_none}")

    sample_text_edge_cases = "Text with url at the end http://example.com"
    candidates_edge = extract_url_candidates(sample_text_edge_cases)
    print(f"\nFound URL candidates in '{sample_text_edge_cases}': {candidates_edge}")

    sample_text_multiple_on_line = "http://first.com and then https://second.com on the same line."
    candidates_multiple = extract_url_candidates(sample_text_multiple_on_line)
    print(f"\nFound URL candidates in '{sample_text_multiple_on_line}': {candidates_multiple}")

    print("\nIdentifying schemes for found candidates from extract_url_candidates:")
    all_extracted_candidates = []
    if candidates: all_extracted_candidates.extend(candidates)
    if candidates_none: all_extracted_candidates.extend(candidates_none) # Should be empty
    if candidates_edge: all_extracted_candidates.extend(candidates_edge)
    if candidates_multiple: all_extracted_candidates.extend(candidates_multiple)

    unique_extracted_candidates = sorted(list(set(all_extracted_candidates))) # Process unique URLs once
    for url in unique_extracted_candidates:
        scheme = identify_url_scheme(url)
        print(f"URL: {url}, Scheme: {scheme}")

    print("\nTesting identify_url_scheme directly with various cases:")
    test_urls_for_scheme = [
        "http://www.example.com",
        "https://example.org/path?query=param",
        "ftp://ftp.example.com/resource",
        "mailto:user@example.com",
        "tel:1234567890",
        "urn:isbn:0451450523",
        "customScheme://data",
        "HTTP://UPPERCASE.SCHEME/PATH", # Uppercase scheme
        "HtTpS://MixedCase.Scheme/path", # Mixed case scheme
        "example.com", # No scheme
        "://missing.scheme.com", # Invalid scheme start (starts with :)
        "//network.path.com/resource", # Network-path reference (scheme-relative)
        "  http://leadingwhitespace.com", # Leading whitespace
        "javascript:void(0);", # Javascript scheme
        "data:text/plain,Hello%2C%20World!", # Data scheme
        "ws://websocket.example.com", # WebSocket scheme
        "wss://secure.websocket.example.com" # Secure WebSocket scheme
    ]
    for test_url in test_urls_for_scheme:
        scheme = identify_url_scheme(test_url)
        print(f"URL: \"{test_url}\", Scheme: \"{scheme}\"")

    print("\nTesting detect_js_scheme with various cases:")
    test_urls_for_js_scheme = [
        ("javascript:alert('XSS')", True),
        ("JAVASCRIPT:alert('XSS')", True),
        ("vbscript:msgbox('XSS')", True),
        ("VBSCRIPT:msgbox('XSS')", True),
        ("  javascript:alert('XSS')  ", True), # Leading/trailing whitespace
        ("data:image/jpeg;base64,...", False),
        ("data:image/png;base64,...", False),
        ("data:text/plain,Hello", False), # Not javascript or vbscript, and not data:image
        ("data:text/html,<script>alert(1)</script>", False), # data: scheme, but not javascript: or vbscript:
        ("http://example.com", False),
        ("https://example.com", False),
        ("ftp://example.com", False),
        ("mailto:user@example.com", False),
        ("example.com", False), # No scheme
        ("javascript://alert('XSS')", True), # Scheme is 'javascript'
        ("vbscript://alert('XSS')", True),   # Scheme is 'vbscript'
        ("", False), # Empty string
        ("    ", False), # Whitespace only string
        ("ジャバスクリプト:alert(1)", False) # Non-ASCII javascript-like scheme name
    ]

    for url, expected_result in test_urls_for_js_scheme:
        result = detect_js_scheme(url)
        status = "Correct" if result == expected_result else f"Incorrect (expected {expected_result})"
        print(f"URL: \"{url}\", Harmful: {result} ({status})")
