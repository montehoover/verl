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


def verify_javascript_protocol(url_input: str) -> bool:
    """
    Checks for potentially harmful URL schemes (javascript, vbscript) in a string,
    while ignoring valid image data URLs.

    Args:
        url_input: The URL string to check.

    Returns:
        True if it contains dangerous URL schemes (and is not a data:image URL),
        False otherwise or if a regex error occurs.
    """
    if not isinstance(url_input, str):
        return False

    normalized_url = url_input.strip().lower()

    try:
        # Check for legitimate image data URLs first
        # Pattern: data:image/(png|jpeg|gif|webp|svg+xml|...)
        data_image_pattern = r'^data:image/'
        if re.match(data_image_pattern, normalized_url):
            return False  # It's a legitimate data image URL

        # Check for dangerous schemes like javascript: or vbscript:
        harmful_scheme_pattern = r'^(javascript|vbscript):'
        if re.search(harmful_scheme_pattern, normalized_url):
            return True  # Harmful scheme found
            
    except re.error:
        # In case of a regex compilation or matching error,
        # treat as not harmful to be safe and avoid crashing.
        return False

    return False  # No harmful scheme found


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

    print("\nVerifying JavaScript protocol:")
    test_urls_for_protocol = [
        ("javascript:alert('XSS')", True),
        ("vbscript:msgbox('XSS')", True),
        ("JavaScript:alert('XSS')", True),
        ("  javascript:alert('XSS')  ", True),
        ("http://example.com", False),
        ("https://example.com", False),
        ("ftp://example.com", False),
        ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA", False),
        ("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD", False),
        ("data:text/plain;charset=utf-8;base64,SGVsbG8sIFdvcmxkIQ==", False),
        ("javascript:data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA", True),
        ("vbscript:data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD", True),
        ("text string", False),
        ("", False),
        (None, False) 
    ]

    for url_test, expected_result in test_urls_for_protocol:
        actual_result = verify_javascript_protocol(url_test)
        # Handle printing for None type if it's part of test_urls_for_protocol
        input_repr = "None" if url_test is None else f'"{url_test}"'
        print(f"Input: {input_repr}, Expected: {expected_result}, Got: {actual_result} -> {'Pass' if actual_result == expected_result else 'Fail'}")
