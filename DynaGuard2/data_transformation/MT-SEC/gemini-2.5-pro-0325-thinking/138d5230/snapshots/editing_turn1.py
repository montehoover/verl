from urllib.parse import urlparse

def extract_domain(url_string):
    """
    Extracts the domain from a URL string.

    Args:
        url_string: The URL string to parse.

    Returns:
        The domain part of the URL, or None if parsing fails or no domain is found.
    """
    try:
        parsed_url = urlparse(url_string)
        return parsed_url.netloc
    except Exception:
        # Handle potential parsing errors, though urlparse is generally robust
        return None

if __name__ == '__main__':
    # Example Usage
    urls_to_test = [
        "http://www.example.com/path/to/page?query=string#fragment",
        "https://subdomain.example.co.uk:8080/another/path",
        "ftp://user:password@example.org/resource",
        "example.com",  # Scheme missing, urlparse might treat this as a path
        "invalid-url-string",
        "http://localhost:3000/test"
    ]

    for url in urls_to_test:
        domain = extract_domain(url)
        print(f"URL: {url} -> Domain: {domain}")

    # Test with a more specific case where netloc might be empty if scheme is missing
    # and urlparse treats it as a path
    print(f"URL: {'justadomain.com/path'} -> Domain: {extract_domain('//justadomain.com/path')}")
    # For "justadomain.com/path", urlparse might return '' for netloc.
    # If you want to handle such cases as "justadomain.com", you might need more logic
    # or ensure URLs are prefixed with a scheme (e.g., by adding // if missing).
    # For now, this function strictly relies on urlparse's netloc.
