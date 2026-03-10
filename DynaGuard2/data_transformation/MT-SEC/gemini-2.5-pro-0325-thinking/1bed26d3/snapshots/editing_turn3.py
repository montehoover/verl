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

def check_javascript_scheme(url_str: str) -> bool:
    """
    Detects potentially harmful URL schemes within a string, such as 'javascript'
    or 'vbscript', while ignoring safe image data URLs.

    Args:
        url_str: The URL string to check.

    Returns:
        True if it contains dangerous URL schemes (excluding legitimate image data URLs),
        False otherwise.
    """
    # Check for legitimate image data URLs first, these are considered safe.
    # We match at the beginning of the string, case-insensitively.
    if re.match(r'data:image/', url_str, re.IGNORECASE):
        return False

    # Check for dangerous schemes like 'javascript:' or 'vbscript:'.
    # We search anywhere in the string, case-insensitively.
    # The pattern looks for 'javascript:' or 'vbscript:'
    try:
        if re.search(r'(javascript|vbscript):', url_str, re.IGNORECASE):
            return True
    except re.error:
        # According to the prompt, we don't explicitly raise exceptions for regex errors.
        # We can log this or handle it silently. For now, assume it means no dangerous scheme found.
        return False
        
    return False

if __name__ == '__main__':
    # ... (previous __main__ content remains) ...

    print("\n--- Testing check_javascript_scheme ---")
    test_urls_for_js_scheme = [
        ("javascript:alert('XSS')", True),
        ("vbscript:msgbox('XSS')", True),
        ("JavaScript:alert('XSS')", True),
        ("VBScript:msgbox('XSS')", True),
        ("http://example.com", False),
        ("https://example.com", False),
        ("ftp://example.com", False),
        ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA", False),
        ("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD", False),
        ("data:text/html;charset=utf-8,javascript:alert(1)", False), # data:image is the key
        ("data:application/javascript;base64,YWxlcnQoMSk=", True), # Not data:image
        ("  javascript:alert('XSS')  ", True), # Leading/trailing spaces
        ("text containing javascript:alert('XSS') inside", True),
        ("text containing VBSCRIPT:code here", True),
        ("text with no scripts", False),
        ("text with javascript but no colon", False),
        ("text with data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs= and then javascript:alert(1)", False), # data:image takes precedence
        ("javascript://%0Aalert(1)", True), # URL encoded newline
        ("vbscript://%0Amsgbox(1)", True),
    ]

    for url_input, expected_result in test_urls_for_js_scheme:
        result = check_javascript_scheme(url_input)
        print(f"URL: '{url_input}', Dangerous: {result}, Expected: {expected_result}, Pass: {result == expected_result}")
