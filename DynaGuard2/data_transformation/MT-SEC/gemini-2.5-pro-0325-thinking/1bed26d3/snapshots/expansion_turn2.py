import re

def extract_components(text: str) -> list:
    """
    Extracts words, numbers, or symbols from the given text using regex.

    Args:
        text: The input string.

    Returns:
        A list of extracted components.
    """
    # \w+ matches one or more word characters (alphanumeric + underscore)
    # \S matches any non-whitespace character (catches symbols)
    pattern = r"\w+|\S"
    components = re.findall(pattern, text)
    return components

def identify_url_scheme(url: str) -> str:
    """
    Identifies the scheme of a given URL using regex.

    Args:
        url: The input URL string.

    Returns:
        The scheme of the URL (e.g., 'http', 'https', 'ftp') or 'no_scheme' if not found.
    """
    # Scheme typically starts with a letter, followed by letters, digits, '+', '.', or '-'
    # and ends with a colon.
    match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):", url)
    if match:
        return match.group(1)
    return "no_scheme"

if __name__ == '__main__':
    sample_text = "Hello, world! This is a test with 123 numbers & symbols like $ and %."
    extracted = extract_components(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Extracted components: {extracted}")

    sample_text_2 = "Another-example_with_various.chars"
    extracted_2 = extract_components(sample_text_2)
    print(f"Original text: '{sample_text_2}'")
    print(f"Extracted components: {extracted_2}")

    sample_text_3 = "  Leading and trailing spaces  "
    extracted_3 = extract_components(sample_text_3)
    print(f"Original text: '{sample_text_3}'")
    print(f"Extracted components: {extracted_3}")

    sample_text_4 = ""
    extracted_4 = extract_components(sample_text_4)
    print(f"Original text: '{sample_text_4}'")
    print(f"Extracted components: {extracted_4}")

    print("\n--- URL Scheme Identification ---")
    urls_to_test = [
        "http://www.example.com",
        "https://example.org/path?query=value#fragment",
        "ftp://ftp.example.net/file.txt",
        "javascript:alert('hello')",
        "mailto:user@example.com",
        "tel:+1-555-555-5555",
        "urn:isbn:0451450523",
        "example.com",
        "://missing.scheme.com",
        "  https://leading.space.com", # scheme should still be found if regex handles it
        "HTTP://UPPERCASE.SCHEME.COM",
        "custom-scheme+v1.0:data"
    ]
    for url_test in urls_to_test:
        scheme = identify_url_scheme(url_test)
        print(f"URL: '{url_test}', Scheme: '{scheme}'")
