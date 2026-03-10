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
    # This pattern looks for http/https, optional www, and then a domain-like structure.
    # It's not exhaustive but covers many common cases.
    url_pattern = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|'  # http:// or https://
        r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+|'  # www.
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}(?:/\S*)?' # domain.tld/path
    )
    return url_pattern.findall(text)

if __name__ == '__main__':
    sample_text_1 = "Visit our website at http://example.com or www.example.org. Also check https://another-example.net/path?query=param."
    sample_text_2 = "No urls here."
    sample_text_3 = "Find me at example.com/page and also at sub.example.co.uk/another/page.html. ftp://files.example.com is not matched by this simple regex."
    sample_text_4 = "Check out google.com and my-site.info/path."

    print(f"URLs in '{sample_text_1}': {find_urls(sample_text_1)}")
    print(f"URLs in '{sample_text_2}': {find_urls(sample_text_2)}")
    print(f"URLs in '{sample_text_3}': {find_urls(sample_text_3)}")
    print(f"URLs in '{sample_text_4}': {find_urls(sample_text_4)}")
