import re

def extract_components(text: str) -> list[str]:
    """
    Extracts words, numbers, and special characters from a text string.

    Args:
        text: The input string.

    Returns:
        A list of extracted components.
    """
    # \w+ matches sequences of word characters (letters, numbers, and underscore)
    # [^\w\s] matches any character that is not a word character and not whitespace (special characters)
    pattern = r"\w+|[^\w\s]"
    components = re.findall(pattern, text)
    return components

def identify_url_scheme(url: str) -> str:
    """
    Identifies the scheme of a given URL string.

    Args:
        url: The URL string.

    Returns:
        The scheme as a string (e.g., 'http', 'https') or 'no_scheme' if not found.
    """
    # Regex to match a URL scheme: starts with a letter, followed by letters, digits, '+', '.', or '-'
    # and ends with a colon.
    scheme_pattern = r"^([a-zA-Z][a-zA-Z0-9+.-]*):"
    match = re.match(scheme_pattern, url)
    if match:
        return match.group(1)  # Return the first capturing group (the scheme itself)
    return "no_scheme"

if __name__ == '__main__':
    sample_text = "Hello World! 123_test - 45.67?"
    extracted = extract_components(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Extracted components: {extracted}")

    sample_text_2 = "test@example.com #tag (comment)"
    extracted_2 = extract_components(sample_text_2)
    print(f"Original text: '{sample_text_2}'")
    print(f"Extracted components: {extracted_2}")

    sample_text_3 = ""
    extracted_3 = extract_components(sample_text_3)
    print(f"Original text: '{sample_text_3}'")
    print(f"Extracted components: {extracted_3}")

    sample_text_4 = "!@#$%^&*()"
    extracted_4 = extract_components(sample_text_4)
    print(f"Original text: '{sample_text_4}'")
    print(f"Extracted components: {extracted_4}")

    print("\n--- Testing identify_url_scheme ---")
    urls_to_test = [
        "http://example.com",
        "https://example.com/path?query=value",
        "ftp://ftp.example.com",
        "javascript:alert('hello')",
        "mailto:user@example.com",
        "custom-scheme://data",
        "example.com", # No scheme
        "://missing.scheme", # Invalid scheme start
        "http", # No colon
        "1http://example.com" # Starts with number
    ]
    for url_test in urls_to_test:
        scheme = identify_url_scheme(url_test)
        print(f"URL: '{url_test}', Scheme: '{scheme}'")
