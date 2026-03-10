import re

def extract_components(text: str) -> list[str]:
    """
    Extracts words, numbers, and special characters from a text string.

    Args:
        text: The input string.

    Returns:
        A list of extracted components (words, numbers, special characters).
    """
    # \w+ matches one or more word characters (letters, numbers, and underscore)
    # [^\w\s] matches any character that is not a word character and not whitespace (i.e., special characters)
    pattern = r"\w+|[^\w\s]"
    components = re.findall(pattern, text)
    return components

def identify_url_scheme(url: str) -> str:
    """
    Identifies the scheme of a given URL.

    Args:
        url: The URL string.

    Returns:
        The scheme of the URL (e.g., 'http', 'https', 'ftp') or 'no_scheme' if not found.
    """
    # Regex to find a scheme at the beginning of the URL
    # A scheme consists of letters, followed by '://'
    # It also handles cases like 'javascript:' or 'mailto:' which don't have '//'
    match = re.match(r"([a-zA-Z][a-zA-Z0-9+.-]*)(:(//)?)?", url)
    if match:
        return match.group(1).lower()
    return "no_scheme"

def verify_javascript_protocol(url_input: str) -> bool:
    """
    Verifies if the URL input contains potentially harmful schemes like 'javascript:' or 'vbscript:',
    while allowing 'data:image/...' schemes.

    Args:
        url_input: The URL string to verify.

    Returns:
        True if a potentially harmful scheme is detected, False otherwise.
    """
    # Convert to lowercase for case-insensitive matching
    lower_url_input = url_input.lower()

    # Check for allowed 'data:image/' scheme first
    if re.match(r"data:image/", lower_url_input):
        return False  # Not harmful

    # Check for harmful schemes
    harmful_schemes_pattern = r"^(javascript|vbscript):"
    if re.match(harmful_schemes_pattern, lower_url_input):
        return True  # Harmful scheme detected

    return False # No harmful scheme detected

if __name__ == '__main__':
    sample_text = "Hello, world! This is a test with 123 numbers & *special* characters."
    extracted = extract_components(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Extracted components: {extracted}")

    sample_text_2 = "Another - example_with_underscore anD 456."
    extracted_2 = extract_components(sample_text_2)
    print(f"Original text: {sample_text_2}")
    print(f"Extracted components: {extracted_2}")

    sample_text_3 = "@#$ %^&"
    extracted_3 = extract_components(sample_text_3)
    print(f"Original text: {sample_text_3}")
    print(f"Extracted components: {extracted_3}")

    sample_text_4 = ""
    extracted_4 = extract_components(sample_text_4)
    print(f"Original text: '{sample_text_4}'")
    print(f"Extracted components: {extracted_4}")

    print("\n--- URL Scheme Identification ---")
    urls_to_test = [
        "http://www.example.com",
        "https://example.org/path?query=value",
        "ftp://files.example.net",
        "javascript:alert('hello')",
        "mailto:user@example.com",
        "tel:+1234567890",
        "urn:isbn:0451450523",
        "example.com",
        "://missing.scheme.com",
        "HTTP://CaseSensitive.Example.COM",
        "HtTpS://MixedCase.Example.ORG",
        "data:text/plain;base64,SGVsbG8sIFdvcmxkIQ=="
    ]
    for url_test in urls_to_test:
        scheme = identify_url_scheme(url_test)
        print(f"URL: '{url_test}', Scheme: '{scheme}'")

    print("\n--- Harmful URL Scheme Verification ---")
    harmful_urls_to_test = [
        "javascript:alert('XSS')",
        "JAVASCRIPT:alert('XSS')",
        "vbscript:msgbox('XSS')",
        "VBSCRIPT:msgbox('XSS')",
        "http://example.com",
        "https://example.com",
        "data:image/jpeg;base64,SGVsbG8sIFdvcmxkIQ==",
        "data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==", # This should be considered harmful by this function's logic
        "  javascript:alert('padded')", # Leading spaces
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
        "ftp://example.com",
        "someotherprotocol:danger", # Not explicitly harmful by current definition
        "javascript : alert('with space')", # Space after scheme
        "vbscript : msgbox('with space')" # Space after scheme
    ]
    for url_test in harmful_urls_to_test:
        is_harmful = verify_javascript_protocol(url_test)
        print(f"URL: '{url_test}', Is Harmful: {is_harmful}")
