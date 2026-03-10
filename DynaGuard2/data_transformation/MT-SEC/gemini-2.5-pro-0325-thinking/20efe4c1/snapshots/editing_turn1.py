import re

def find_urls(text: str) -> list[str]:
    """
    Scans a string and extracts all the URL-like patterns it contains.

    Args:
        text: The string to scan.

    Returns:
        A list of all the URL-like substrings found in the text.
    """
    # A common regex for matching URLs. This can be adjusted for more specific needs.
    # This regex looks for http/https/ftp, optional www, domain name, and path/query parameters.
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)

if __name__ == '__main__':
    sample_text_with_urls = """
    Visit our website at https://www.example.com for more information.
    You can also check out http://example.org/about or ftp://files.example.net/data.
    Some invalid urls like http://localhost or just www.google.com might not be caught perfectly
    or might be caught depending on the regex strictness.
    Another one: https://sub.domain.example.co.uk/path?query=value#fragment
    """
    urls_found = find_urls(sample_text_with_urls)
    print("URLs found:")
    for url in urls_found:
        print(url)

    sample_text_without_urls = "This is a string with no URLs."
    urls_not_found = find_urls(sample_text_without_urls)
    print("\nURLs found in text without URLs:")
    for url in urls_not_found:
        print(url)
    if not urls_not_found:
        print("(No URLs found)")
