import re

def find_urls(text: str) -> list[str]:
    """
    Scans a string and extracts all URL-like patterns.

    Args:
        text: The string to scan.

    Returns:
        A list of URL-like substrings found in the text.
    """
    # A simple regex to find URL-like patterns.
    # This pattern looks for http/https followed by non-whitespace characters.
    # It's a basic pattern and might not cover all URL variations or edge cases.
    url_pattern = r"https?://\S+"
    urls = re.findall(url_pattern, text)
    return urls

if __name__ == '__main__':
    sample_text_with_urls = "Visit our website at http://example.com or check out https://www.another-example.org/path?query=param. Also, ftp://fileserver.com is not matched by this simple regex."
    found_urls = find_urls(sample_text_with_urls)
    print(f"Found URLs: {found_urls}")

    sample_text_without_urls = "This is a string with no URLs."
    found_urls_none = find_urls(sample_text_without_urls)
    print(f"Found URLs: {found_urls_none}")

    sample_text_multiple_urls = "Check these: http://first.com, https://second.net/page, and http://third.org."
    found_urls_multiple = find_urls(sample_text_multiple_urls)
    print(f"Found URLs: {found_urls_multiple}")
