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

def detect_js_scheme(url_string: str) -> bool:
    """
    Detects potentially harmful URL schemes like 'javascript' or 'vbscript',
    while ignoring valid image data URLs.

    Args:
        url_string: The string to check.

    Returns:
        True if a harmful scheme is detected (and it's not an image data URL),
        False otherwise or if a regex error occurs.
    """
    try:
        # Pattern for 'javascript:' or 'vbscript:' schemes, allowing leading whitespace.
        # Case-insensitive.
        harmful_scheme_pattern = re.compile(r'^\s*(javascript|vbscript):', re.IGNORECASE)

        # Pattern for 'data:image/...' schemes, allowing leading whitespace.
        # Case-insensitive. Covers common image types like png, jpeg, gif, webp, svg.
        image_data_scheme_pattern = re.compile(
            r'^\s*data:image/(?:png|jpeg|gif|webp|svg\+xml)', re.IGNORECASE
        )

        # First, check if it's a legitimate image data URL. If so, it's not considered harmful.
        if image_data_scheme_pattern.match(url_string):
            return False

        # Then, check for harmful schemes.
        if harmful_scheme_pattern.match(url_string):
            return True

    except re.error:
        # In case of a regex compilation or matching error,
        # treat as non-harmful as per "we won't explicitly raise exceptions".
        return False

    return False

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

    print("\n--- Testing detect_js_scheme ---")
    test_cases_for_detect_js_scheme = [
        # Harmful schemes
        ("javascript:alert('XSS')", True),
        ("vbscript:evil()", True),
        ("JAVASCRIPT:console.log(1)", True), # Case-insensitive scheme
        ("  javascript:alert('Padded with spaces at start')", True), # Leading whitespace before scheme
        ("javascript://alert('XSS')", True), # Scheme is 'javascript', rest is payload

        # Safe schemes or not a scheme we're looking for
        ("http://example.com", False),
        ("https://example.com", False),
        ("ftp://example.com", False),
        ("mailto:user@example.com", False),
        ("tel:1234567890", False),

        # Valid image data URLs (should be ignored, return False)
        ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA", False),
        ("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD", False),
        ("data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7", False),
        ("data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==", False),
        ("data:image/svg+xml;charset=utf-8,%3Csvg%3E%3C/svg%3E", False),
        ("  data:image/png;base64,abc...", False), # Leading whitespace before data scheme

        # Data URLs that are not images (should be False as their scheme is 'data', not 'javascript'/'vbscript')
        ("data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==", False),
        ("data:application/json;base64,e30=", False),
        ("data:application/javascript;base64,YWxlcnQoMSk=", False), # Scheme is 'data'

        # Malformed, other non-matching strings, or strings where harmful scheme is not at the start
        ("javascript :alert('XSS with space before colon')", False), # Space before colon is not standard for scheme
        ("vbscript :evil()", False), # Space before colon
        ("javascript", False), # Missing colon
        ("example.com", False),
        ("", False), # Empty string
        ("    ", False), # Whitespace only string
        ("text before javascript:alert(0)", False), # Scheme must be at the start (after optional whitespace)
        ("http://example.com#javascript:void(0)", False), # JS is in fragment, not scheme
    ]

    all_detect_js_scheme_tests_passed = True
    print("Running detect_js_scheme tests:")
    for i, (url_str, expected) in enumerate(test_cases_for_detect_js_scheme):
        result = detect_js_scheme(url_str)
        is_match = (result == expected)
        
        printable_url = url_str[:70] + '...' if len(url_str) > 70 else url_str
        status_msg = 'Pass' if is_match else 'FAIL'
        print(f"  Test {i+1:02d}: Expected {str(expected):<5} | Got {str(result):<5} | URL: '{printable_url}' -> {status_msg}")
        
        if not is_match:
            all_detect_js_scheme_tests_passed = False
            # More detailed log for failures can be helpful during debugging
            # print(f"    [FAIL Detail] URL: '{url_str}', Expected: {expected}, Got: {result}")

    if all_detect_js_scheme_tests_passed:
        print("All detect_js_scheme tests PASSED successfully.")
    else:
        print("!!! Some detect_js_scheme tests FAILED. Please review the output above. !!!")
