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

def analyze_scheme(url: str) -> bool:
    """
    Detects potentially harmful URL schemes, ignoring valid image data URLs.

    Args:
        url: The URL string to analyze.

    Returns:
        True if a potentially harmful scheme is detected, False otherwise.
    """
    # Extract scheme using regex. Scheme is group 1.
    # Pattern: starts with letters, followed by letters, digits, '+', '.', or '-'
    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):", url)
    if not scheme_match:
        return False  # No scheme found or malformed URL start

    scheme = scheme_match.group(1).lower()

    harmful_explicit_schemes = {'javascript', 'vbscript'}
    if scheme in harmful_explicit_schemes:
        return True

    if scheme == 'data':
        # Check if it's a known safe image data URL.
        # Pattern: "data:image/(png|jpeg|gif|webp|svg+xml)"
        # followed by optional parameters (like ;base64) and then a comma.
        # re.IGNORECASE handles "DATA:IMAGE/PNG..."
        if re.match(r"data:image/(?:png|jpeg|gif|webp|svg\+xml)(?:;[^,]*)?,", url, re.IGNORECASE):
            return False  # Safe image data URL
        else:
            # Other data URLs (e.g., data:text/html) are considered potentially harmful
            return True

    # All other schemes (http, https, ftp, etc.) are considered not harmful by this function
    return False

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

    print("\nAnalyzing URL Schemes for potential harm:")
    test_urls_for_scheme_analysis = [
        "javascript:alert('XSS')",
        "vbscript:msgbox('XSS')",
        "http://example.com",
        "https://example.com",
        "ftp://example.com/file.txt",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
        "DATA:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD",
        "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
        "data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==",
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3C/svg%3E",
        "data:image/svg+xml;charset=utf-8,%3Csvg%3E%3C/svg%3E",
        "data:text/html,<script>alert('harmful')</script>",
        "data:application/json,{\"key\":\"value\"}",
        "tel:+1234567890",
        "mailto:user@example.com",
        "customscheme:someaction",
        "JAVASCRIPT:alert('UPPERCASE XSS')",
        "vbscript:(alert('parentheses'))", # VBScript often has parentheses
        "  javascript:alert('leading spaces')", # Leading spaces might be trimmed by browsers
        "data:image/png,sometrailingstuff", # Valid image data start
        "data:image/evil;base64,xxxxxxxx", # Unknown image subtype, potentially harmful under 'data'
        "data:,simpletext", # Generic data URL, potentially harmful
        "blob:http://example.com/uuid-goes-here", # Blob URLs, typically not harmful by scheme alone
        "filesystem:http://example.com/temporary/file.txt", # Filesystem URLs
        "thisisnotascheme:foo", # Valid scheme format, but not on harmful list
        "invalid-scheme başlangıcı:", # Not a valid scheme start
        "http:javascript:alert('nested?')", # Outer scheme is http
        " javascript:alert('test')", # Scheme with leading space, current regex won't catch scheme
    ]

    for t_url in test_urls_for_scheme_analysis:
        is_harmful = analyze_scheme(t_url.strip()) # Test with stripped URL as browsers might trim
        print(f"URL: \"{t_url}\" -> Harmful: {is_harmful}")

    # Test case for URL with leading spaces if analyze_scheme is expected to handle it
    # (current regex for scheme extraction `^([a-zA-Z]...` won't match if there are leading spaces)
    # If stripping is done outside, then the last test case " javascript:..." will be handled by t_url.strip()
    # If analyze_scheme itself should strip, it needs modification. The current test uses t_url.strip().
