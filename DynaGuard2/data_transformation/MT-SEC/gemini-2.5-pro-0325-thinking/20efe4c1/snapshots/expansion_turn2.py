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
