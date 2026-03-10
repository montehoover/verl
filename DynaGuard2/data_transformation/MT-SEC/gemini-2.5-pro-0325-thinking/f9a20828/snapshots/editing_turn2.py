import re

def extract_urls(text: str) -> list[tuple[str, str]]:
    """
    Extracts all URLs from a given string and identifies their schemes.

    Args:
        text: The string to parse for URLs.

    Returns:
        A list of tuples, where each tuple contains the URL and its scheme.
        For example: [('https://www.example.com', 'https'), ...]
    """
    # Regex to match URLs and capture the scheme.
    # Scheme: according to RFC 3986 (ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ))
    # This regex matches common schemes like http, https, ftp, file, etc.
    # followed by :// and the rest of the URL.
    url_pattern = re.compile(
        r'([a-zA-Z][a-zA-Z0-9+.-]*)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    matches = url_pattern.finditer(text)
    urls_with_schemes = []
    for match in matches:
        full_url = match.group(0)
        scheme = match.group(1)
        urls_with_schemes.append((full_url, scheme))
    return urls_with_schemes

if __name__ == '__main__':
    sample_text_with_urls = """
    Welcome to our website! You can find us at https://www.example.com.
    For more information, please visit http://example.org/info.
    Check out our new blog post at https://blog.example.com/new-post?id=123#comments.
    This is not a url: example.com. But this is: ftp://files.example.com/data.zip.
    Another one: http://localhost:8000/path
    """
    extracted = extract_urls(sample_text_with_urls)
    print("Extracted URLs and Schemes:")
    for url, scheme in extracted:
        print(f"URL: {url}, Scheme: {scheme}")

    sample_text_without_urls = "This is a string with no URLs."
    extracted_none = extract_urls(sample_text_without_urls)
    print("\nExtracted URLs and Schemes from text without URLs:")
    if not extracted_none:
        print("No URLs found.")
    else:
        for url, scheme in extracted_none: # Should be empty, but good practice to loop
            print(f"URL: {url}, Scheme: {scheme}")

    sample_text_edge_cases = "Text with url at the end https://example.com/end. And one at start: http://start.example.com and one in middle http://middle.example.com."
    extracted_edge = extract_urls(sample_text_edge_cases)
    print("\nExtracted URLs and Schemes from edge case text:")
    for url, scheme in extracted_edge:
        print(f"URL: {url}, Scheme: {scheme}")
