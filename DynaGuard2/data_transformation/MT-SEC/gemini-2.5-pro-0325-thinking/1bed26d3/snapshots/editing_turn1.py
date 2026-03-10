import re

def find_urls(text: str) -> list[str]:
    """
    Scans a string and extracts all URL-like patterns it contains.

    Args:
        text: The string to scan for URLs.

    Returns:
        A list of URL-like substrings found within the text.
    """
    # A common regex for matching URLs. This can be adjusted for more specific needs.
    # This pattern looks for http/https/ftp, optional www, domain name, and path/query parameters.
    url_pattern = re.compile(
        r'http[s]?://'  # http:// or https://
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # domain name and path
    )
    return url_pattern.findall(text)

if __name__ == '__main__':
    sample_text_with_urls = """
    Visit our website at http://www.example.com for more information.
    You can also check out https://example.org/path?query=param.
    Another link is ftp://files.example.net/data.txt.
    This is not a url: www.missingprotocol.com
    And another one: http://localhost:8000/my/page
    Check this: https://sub.domain.example.co.uk/another/path.html#fragment
    """
    urls_found = find_urls(sample_text_with_urls)
    print("URLs found:")
    for url in urls_found:
        print(url)

    sample_text_without_urls = "This is a string with no URLs."
    urls_found_none = find_urls(sample_text_without_urls)
    print("\nURLs found in text without URLs:")
    print(urls_found_none)

    sample_text_edge_cases = "Text with url http://example.com.And anotherhttps://another.com immediately after."
    urls_found_edge = find_urls(sample_text_edge_cases)
    print("\nURLs found in edge case text:")
    for url in urls_found_edge:
        print(url)
