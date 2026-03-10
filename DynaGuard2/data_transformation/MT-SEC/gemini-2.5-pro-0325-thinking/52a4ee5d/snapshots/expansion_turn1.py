import re

def extract_url_candidates(text: str) -> list[str]:
    """
    Finds and returns a list of URL-like patterns within the given text.

    Args:
        text: The input string to search for URLs.

    Returns:
        A list of strings, where each string is a potential URL.
    """
    # A common, but not exhaustive, regex for URLs.
    # This regex looks for http/https, optional www, domain name, and path/query parameters.
    url_pattern = re.compile(
        r'http[s]?://'  # http:// or https://
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # domain name and path
    )
    return url_pattern.findall(text)

if __name__ == '__main__':
    sample_text_with_urls = """
    Visit our website at http://www.example.com for more information.
    You can also check out https://example.org/path?query=param.
    Another link is http://sub.example.co.uk/another/path.
    Not a url: example.com. And this ftp://old.server.com is not matched by this regex.
    """
    candidates = extract_url_candidates(sample_text_with_urls)
    print("Found URL candidates:")
    for url in candidates:
        print(url)

    sample_text_without_urls = "This is a string with no URLs."
    candidates_none = extract_url_candidates(sample_text_without_urls)
    print(f"\nFound URL candidates in '{sample_text_without_urls}': {candidates_none}")

    sample_text_edge_cases = "Text with url at the end http://example.com"
    candidates_edge = extract_url_candidates(sample_text_edge_cases)
    print(f"\nFound URL candidates in '{sample_text_edge_cases}': {candidates_edge}")

    sample_text_multiple_on_line = "http://first.com and then https://second.com on the same line."
    candidates_multiple = extract_url_candidates(sample_text_multiple_on_line)
    print(f"\nFound URL candidates in '{sample_text_multiple_on_line}': {candidates_multiple}")
