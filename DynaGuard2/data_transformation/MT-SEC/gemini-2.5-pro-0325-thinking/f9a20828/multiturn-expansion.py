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


def analyze_scheme(url: str) -> bool:
    """
    Analyzes a URL string to determine if it includes potentially harmful schemes,
    while allowing specific safe 'data:' schemes like for images.

    Args:
        url: The input URL string.

    Returns:
        True if a potentially harmful scheme is detected, False otherwise.
    """
    normalized_url = url.strip().lower()

    # Allow data:image/(png|jpeg|gif|webp|svg+xml) schemes
    # Regex covers common image types and optional parameters like ;base64, followed by the data itself
    if re.match(r'^data:image/(?:png|jpeg|gif|webp|svg\+xml)(?:;[^,]*)?,', normalized_url):
        return False  # Safe data URI for images

    # Check for explicitly harmful schemes (case-insensitive check due to normalized_url)
    harmful_schemes_prefixes = [
        'javascript:',
        'vbscript:',
        # Add other schemes like 'blob:', 'filesystem:' if they are considered harmful
        # in the specific context where this function will be used.
        # 'file:' can also be sensitive depending on the environment.
    ]

    for prefix in harmful_schemes_prefixes:
        if normalized_url.startswith(prefix):
            return True  # Harmful scheme detected

    # If the URL is not an allowed data:image URI and not an explicitly harmful scheme,
    # it's considered not harmful by this function's current logic.
    # This means 'data:text/html,...' would return False.
    return False  # No explicitly listed harmful scheme found

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

    # Test cases for analyze_scheme
    print("\nTesting analyze_scheme:")
    harmful_test_urls = {
        "javascript:alert(1)": True,
        " JAVASCRIPT:alert('XSS') ": True,
        "vbscript:MsgBox(\"evil\")": True,
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA": False,
        "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4QAiRXhpZgAATU0AKgAAAAgAAQESAAMAAAABAAYAAAA": False,
        "data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3C/svg%3E": False,
        "data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==": False,
        "http://example.com": False,
        "https://example.com": False,
        "ftp://example.com": False,
        "data:text/html,<script>alert(1)</script>": False, # Not 'data:image' and not in harmful_schemes_prefixes
        "data:text/plain,Hello": False, # Same reasoning as above
        "  vbscript:window.location='http://evil.com'  ": True,
        "DATA:IMAGE/GIF;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7": False,
        "file:///etc/passwd": False, # 'file:' is not in harmful_schemes_prefixes by default
        "randomstring": False,
        "": False,
    }

    for url, expected_harmful in harmful_test_urls.items():
        is_harmful = analyze_scheme(url)
        status = "Correct" if is_harmful == expected_harmful else f"Incorrect (expected {expected_harmful})"
        print(f"URL: '{url}', Harmful: {is_harmful} ({status})")

