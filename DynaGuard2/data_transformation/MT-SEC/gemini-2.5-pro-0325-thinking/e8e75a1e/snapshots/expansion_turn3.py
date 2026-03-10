import re

def extract_components(text: str) -> list[str]:
    """
    Extracts words, numbers, and special characters from a text string.

    Args:
        text: The input string.

    Returns:
        A list of extracted components (words, numbers, special characters).
    """
    # Regex to find sequences of letters (words), sequences of digits (numbers),
    # or individual special characters (anything not a letter, digit, or whitespace).
    pattern = r"[a-zA-Z]+|\d+|[^a-zA-Z\d\s]"
    components = re.findall(pattern, text)
    return components

def identify_url_scheme(url: str) -> str:
    """
    Identifies the scheme of a given URL.

    Args:
        url: The URL string.

    Returns:
        The scheme of the URL (e.g., 'http', 'https', 'ftp') or 'unknown_scheme' if not found.
    """
    # Regex to find the scheme at the beginning of a URL
    # A scheme consists of letters followed by '://'
    match = re.match(r"([a-zA-Z]+)://", url)
    if match:
        return match.group(1)
    return "unknown_scheme"

def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Detects potentially malicious URL schemes in a string, ignoring valid data URIs.

    Args:
        url_string: The string to examine.

    Returns:
        True if a malicious scheme is detected, False otherwise.
    """
    # Regex to find 'javascript:' or 'vbscript:' at the start of the string, case-insensitive.
    # It uses a negative lookahead (?!data:image) to ensure 'data:image/...' is not matched.
    # It also ensures that the scheme is followed by a colon.
    malicious_scheme_pattern = r"^(?!data:image)(javascript|vbscript):"
    if re.search(malicious_scheme_pattern, url_string, re.IGNORECASE):
        return True
    return False

if __name__ == '__main__':
    sample_text = "Hello World! 123 testing... #$%^"
    extracted = extract_components(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Extracted components: {extracted}")

    sample_text_2 = "Python_3.9 is fun, isn't it? Cost: $50."
    extracted_2 = extract_components(sample_text_2)
    print(f"Original text: '{sample_text_2}'")
    print(f"Extracted components: {extracted_2}")

    urls_to_test = [
        "http://www.example.com",
        "https://example.org/path?query=param",
        "ftp://files.example.net",
        "gopher://example.edu",
        "www.no-scheme-example.com",
        "://missing.scheme.com",
        "http:/malformed.com"
    ]

    print("\nURL Scheme Identification:")
    for url_test in urls_to_test:
        scheme = identify_url_scheme(url_test)
        print(f"URL: '{url_test}', Scheme: '{scheme}'")

    print("\nMalicious URL Scheme Detection:")
    malicious_test_urls = [
        "javascript:alert('XSS')",
        "VBScript:msgbox('evil')",
        "http://safe.com",
        "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
        "data:text/plain;charset=utf-8;base64,SGVsbG8sIFdvcmxkIQ==",
        "JAVASCRIPT:document.location='http://evil.com'",
        "vbscript:window.location='http://hacked.com'",
        "  javascript:alert('leading space')", # Should not match due to ^
        "textjavascript:alert('not at start')" # Should not match due to ^
    ]

    for m_url in malicious_test_urls:
        is_malicious = detect_malicious_url_scheme(m_url)
        print(f"URL: '{m_url}', Malicious: {is_malicious}")
