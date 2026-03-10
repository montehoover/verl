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
