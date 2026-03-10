import re

def extract_components(text: str) -> dict:
    """
    Extracts words, numbers, and special characters from a given text string.

    Args:
        text: The input string to process.

    Returns:
        A dictionary with keys 'words', 'numbers', and 'special_chars',
        each containing a list of the respective components found in the text.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    numbers = re.findall(r'\b\d+\b', text)
    
    # To find special characters, we can find anything that is not a word character (alphanumeric) or whitespace.
    # We will iterate through the text and identify characters that are not alphanumeric and not whitespace.
    # A simpler approach for special characters might be to find all non-alphanumeric characters, excluding whitespace.
    # However, re.findall(r'[^\w\s]', text) might be too broad or too narrow depending on definition.
    # Let's refine the special characters definition: anything not a letter, not a digit, and not whitespace.
    
    # First, remove all words and numbers from the text to isolate special characters and spaces
    text_without_words_and_numbers = re.sub(r'\b[a-zA-Z]+\b', '', text)
    text_without_words_and_numbers = re.sub(r'\b\d+\b', '', text_without_words_and_numbers)
    
    # Then, find all non-whitespace characters from the remaining string
    special_chars = re.findall(r'[^\w\s]', text) # A common way to find special characters

    return {
        'words': words,
        'numbers': numbers,
        'special_chars': special_chars
    }

if __name__ == '__main__':
    sample_text = "Hello World! 123 testing... #$%^"
    components = extract_components(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Extracted components: {components}")

    sample_text_2 = "Python 3.9 is amazing & fun!"
    components_2 = extract_components(sample_text_2)
    print(f"Original text: {sample_text_2}")
    print(f"Extracted components: {components_2}")
    
    sample_text_3 = "No_special_chars_here_123"
    components_3 = extract_components(sample_text_3)
    print(f"Original text: {sample_text_3}") # Note: Underscore is a word character by \w
    print(f"Extracted components: {components_3}")

    sample_text_4 = "@#$ %^&"
    components_4 = extract_components(sample_text_4)
    print(f"Original text: {sample_text_4}")
    print(f"Extracted components: {components_4}")


def get_url_scheme(url: str) -> str:
    """
    Extracts the scheme from a given URL string.

    Args:
        url: The input URL string.

    Returns:
        The scheme of the URL (e.g., 'http', 'https', 'ftp') or 'no_scheme'
        if no scheme is found.
    """
    match = re.match(r'^([a-zA-Z]+)://', url)
    if match:
        return match.group(1)
    return 'no_scheme'

if __name__ == '__main__':
    # ... (previous main content remains)

    # Test cases for get_url_scheme
    urls_to_test = [
        "https://www.example.com",
        "http://example.com/path?query=value",
        "ftp://files.example.com",
        "gopher://example.com",
        "mailto:user@example.com", # mailto is a scheme, but doesn't always have ://
        "example.com",
        "://invalid.com",
        "ws://localhost:8080",
        "//cdn.example.com/file.js" # protocol-relative URL
    ]

    print("\nTesting get_url_scheme:")
    for test_url in urls_to_test:
        scheme = get_url_scheme(test_url)
        print(f"URL: '{test_url}', Scheme: '{scheme}'")

    # Adjusting mailto test as per common URL scheme definition (alphanum followed by ://)
    # The current regex `^([a-zA-Z]+)://` will not match "mailto:user@example.com"
    # It will correctly identify "mailto" if it were "mailto://user@example.com"
    # For "mailto:user@example.com", it will return 'no_scheme', which is correct for the regex.
    # If a broader definition of scheme (any alphanumeric followed by just ':') is needed, the regex would change.
    # For now, sticking to schemes followed by '://'.

    # Test for protocol-relative URL
    # "//cdn.example.com/file.js" should return 'no_scheme' as it starts with // not scheme://
