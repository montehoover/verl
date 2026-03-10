import re

def extract_components(text: str) -> list:
    """
    Extracts words, numbers, and special characters from a text string.

    Args:
        text: The input string.

    Returns:
        A list of extracted components (words, numbers, special characters).
    """
    # Regex to find words (sequences of letters), numbers (sequences of digits),
    # or any single non-whitespace character that isn't a letter or digit (special characters).
    pattern = re.compile(r"([a-zA-Z]+|\d+|[^a-zA-Z\d\s])")
    components = pattern.findall(text)
    return components

def identify_url_scheme(url: str) -> str:
    """
    Identifies the scheme of a given URL.

    Args:
        url: The URL string.

    Returns:
        The scheme of the URL (e.g., 'http', 'https', 'ftp') or 'no_scheme' if not found.
    """
    # Regex to find the scheme at the beginning of a URL
    # It looks for a sequence of letters followed by '://'
    scheme_pattern = re.compile(r"^([a-zA-Z]+)://")
    match = scheme_pattern.match(url)
    if match:
        return match.group(1)
    return "no_scheme"

def has_script_scheme(url: str) -> bool:
    """
    Checks if a URL string contains potentially harmful script schemes like 'javascript:' or 'vbscript:',
    while ignoring valid 'data:image/...' schemes.

    Args:
        url: The URL string to examine.

    Returns:
        True if a harmful script scheme is present, False otherwise.
    """
    lower_url = url.lower()
    # Ignore valid data:image URLs
    if re.match(r"^data:image/", lower_url):
        return False
    # Check for harmful schemes like javascript: or vbscript:
    # The regex looks for 'javascript' or 'vbscript' followed by a colon, at the start of the string.
    harmful_scheme_pattern = re.compile(r"^(javascript|vbscript):")
    if harmful_scheme_pattern.match(lower_url):
        return True
    return False

if __name__ == '__main__':
    sample_text = "Hello World! 123 testing... #tag @mention"
    extracted = extract_components(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Extracted components: {extracted}")

    sample_text_2 = "Python 3.9 is great, isn't it?"
    extracted_2 = extract_components(sample_text_2)
    print(f"Original text: '{sample_text_2}'")
    print(f"Extracted components: {extracted_2}")

    sample_text_3 = "$%^&*()"
    extracted_3 = extract_components(sample_text_3)
    print(f"Original text: '{sample_text_3}'")
    print(f"Extracted components: {extracted_3}")

    sample_text_4 = "NoSpecialCharactersOrNumbers"
    extracted_4 = extract_components(sample_text_4)
    print(f"Original text: '{sample_text_4}'")
    print(f"Extracted components: {extracted_4}")

    sample_text_5 = "12345"
    extracted_5 = extract_components(sample_text_5)
    print(f"Original text: '{sample_text_5}'")
    print(f"Extracted components: {extracted_5}")

    print("\n--- URL Scheme Identification ---")
    urls_to_test = [
        "http://www.example.com",
        "https://example.org/path?query=param",
        "ftp://files.example.net",
        "example.com",
        "://missing.scheme.com",
        "mailto:user@example.com", # mailto is a scheme, but pattern expects ://
        "git://github.com/user/repo.git"
    ]
    for u in urls_to_test:
        scheme = identify_url_scheme(u)
        print(f"URL: '{u}', Scheme: '{scheme}'")

    print("\n--- Harmful Scheme Detection ---")
    script_urls_to_test = [
        "javascript:alert('XSS')",
        "JAVASCRIPT:alert('XSS')",
        "vbscript:msgbox('XSS')",
        "VBSCRIPT:msgbox('XSS')",
        "http://example.com",
        "https://example.com",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
        "data:text/html,<script>alert('XSS')</script>", # This is not data:image, so should be considered potentially harmful if not handled by other means
        " javascript:alert('leading space')", # Current regex ^ will not match this, which is fine.
        "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
        "ftp://example.com"
    ]
    for u_script in script_urls_to_test:
        is_harmful = has_script_scheme(u_script)
        print(f"URL: '{u_script}', Has harmful script scheme: {is_harmful}")
