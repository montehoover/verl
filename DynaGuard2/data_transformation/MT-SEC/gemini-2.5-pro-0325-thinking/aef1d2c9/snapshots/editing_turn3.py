import re
from collections import defaultdict

def categorize_urls(text: str) -> dict[str, list[str]]:
    """
    Scans a string, identifies all well-formed URLs, and categorizes them by scheme.

    Args:
        text: The string to scan for URLs.

    Returns:
        A dictionary where keys are schemes (e.g., 'http', 'https')
        and values are lists of URLs with that scheme.
    """
    # Regex to match URLs with http, https, ftp, or file schemes.
    # It matches the scheme, '://', and then a simplified pattern for the rest of the URL.
    url_pattern = re.compile(
        r'(?:https?|ftp|file)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    found_urls = re.findall(url_pattern, text)
    
    categorized_urls = defaultdict(list)
    for url in found_urls:
        scheme = url.split("://", 1)[0]
        categorized_urls[scheme].append(url)
            
    return dict(categorized_urls) # Convert defaultdict to dict for cleaner output if preferred

def has_script_scheme(url: str) -> bool:
    """
    Identifies if a URL string contains a risky script scheme (e.g., 'javascript', 'vbscript'),
    while ignoring legitimate image data URLs.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL has a risky script scheme, False otherwise.
    """
    # Convert URL to lowercase for case-insensitive matching
    lower_url = url.lower()

    # Check for legitimate image data URLs first and consider them safe
    if lower_url.startswith("data:image"):
        return False

    # Regex to find risky script schemes like 'javascript:' or 'vbscript:'
    # We match at the beginning of the string (^) after converting to lowercase.
    # The pattern looks for 'javascript:' or 'vbscript:'.
    script_scheme_pattern = re.compile(r"^(javascript|vbscript):")
    
    try:
        if script_scheme_pattern.match(lower_url):
            return True
    except re.error:
        # According to the request, we don't explicitly raise exceptions for re errors.
        # We can log this or handle it silently. For now, assume it's not a match.
        pass
        
    return False

if __name__ == '__main__':
    sample_text_with_urls = """
    Welcome to our website! You can find us at http://www.example.com.
    For more information, please visit https://example.org/about?query=test#section.
    Check out this ftp link: ftp://files.example.net/data.
    Also, a local file link file:///C:/Users/docs/file.txt might be present.
    Invalid URL: www.onlydomain.com or just example.com
    Another valid one: http://localhost:8080/path
    A complex one: https://sub.domain.co.uk:8000/path/to/resource?name=value&type=1#fragment-id
    """
    categorized_urls_result = categorize_urls(sample_text_with_urls)
    if categorized_urls_result:
        print("Categorized URLs:")
        for scheme, urls in categorized_urls_result.items():
            print(f"\nScheme: {scheme}")
            for url in urls:
                print(url)
    else:
        print("No URLs found.")

    sample_text_without_urls = "This is a string with no URLs."
    categorized_urls_none = categorize_urls(sample_text_without_urls)
    if categorized_urls_none:
        print("\nCategorized URLs (second sample):") # Should not print if logic is correct
        for scheme, urls in categorized_urls_none.items():
            print(f"\nScheme: {scheme}")
            for url in urls:
                print(url)
    else:
        print("\nNo URLs found in the second sample.")

    print("\n--- Testing has_script_scheme ---")
    test_urls_for_script_scheme = [
        "javascript:alert('XSS')",
        "JAVASCRIPT:alert('XSS')",
        "vbscript:msgbox('XSS')",
        "VBSCRIPT:msgbox('XSS')",
        "http://example.com",
        "https://example.com",
        "ftp://example.com",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
        "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD",
        "data:text/plain;charset=utf-8;base64,SGVsbG8sIFdvcmxkIQ==",
        " javascript:void(0)", # Leading space, should be handled by strip or careful regex
        "vbscript :evil()", # Space after scheme
        "data:", # Incomplete data URI
        "javascript : alert(1)" # Space after colon
    ]

    for test_url in test_urls_for_script_scheme:
        # To handle leading/trailing spaces in the URL string itself before scheme check
        is_risky = has_script_scheme(test_url.strip())
        print(f"URL: '{test_url}' -> Risky: {is_risky}")
