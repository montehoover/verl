import re

def find_urls(text: str) -> list[str]:
    """
    Scans a string and identifies all well-formed URLs within it.

    Args:
        text: The string to scan for URLs.

    Returns:
        A list of URLs found in the text.
    """
    # A common regex for matching URLs. This regex can be quite complex
    # to cover all valid URL forms, but here's a reasonably comprehensive one.
    # It matches http, https, ftp, and file schemes, optional www,
    # domain name, optional port, path, query parameters, and fragment.
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    urls = re.findall(url_pattern, text)
    return urls

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
    found_urls = find_urls(sample_text_with_urls)
    if found_urls:
        print("Found URLs:")
        for url in found_urls:
            print(url)
    else:
        print("No URLs found.")

    sample_text_without_urls = "This is a string with no URLs."
    found_urls_none = find_urls(sample_text_without_urls)
    if found_urls_none:
        print("\nFound URLs:")
        for url in found_urls_none:
            print(url)
    else:
        print("\nNo URLs found in the second sample.")
