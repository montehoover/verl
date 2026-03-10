import re

def extract_urls(text: str) -> list[str]:
    """
    Extracts all URLs from a given string.

    Args:
        text: The string to parse for URLs.

    Returns:
        A list of URLs found in the text.
    """
    # A common regex for matching URLs. This can be adjusted for more specific needs.
    # This regex matches http, https, ftp, and file schemes, followed by ://
    # then domain name, and optional path, query parameters, and fragment.
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    urls = re.findall(url_pattern, text)
    return urls

if __name__ == '__main__':
    sample_text_with_urls = """
    Welcome to our website! You can find us at https://www.example.com.
    For more information, please visit http://example.org/info.
    Check out our new blog post at https://blog.example.com/new-post?id=123#comments.
    This is not a url: example.com. But this is: ftp://files.example.com/data.zip.
    Another one: http://localhost:8000/path
    """
    extracted = extract_urls(sample_text_with_urls)
    print("Extracted URLs:")
    for url in extracted:
        print(url)

    sample_text_without_urls = "This is a string with no URLs."
    extracted_none = extract_urls(sample_text_without_urls)
    print("\nExtracted URLs from text without URLs:")
    print(extracted_none)

    sample_text_edge_cases = "Text with url at the end https://example.com/end. And one at start: http://start.example.com and one in middle http://middle.example.com."
    extracted_edge = extract_urls(sample_text_edge_cases)
    print("\nExtracted URLs from edge case text:")
    for url in extracted_edge:
        print(url)
