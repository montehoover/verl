from urllib.parse import urlparse

def handle_webhook_url(url_string):
    """
    Parses a webhook URL and returns its domain and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A tuple containing the domain (netloc) and path of the URL.
        Returns (None, None) if the URL cannot be parsed.
    """
    try:
        parsed_url = urlparse(url_string)
        domain = parsed_url.netloc
        path = parsed_url.path
        return domain, path
    except Exception:
        # Handle potential parsing errors, though urlparse is quite robust
        return None, None

if __name__ == '__main__':
    # Example usage:
    test_url_1 = "https://example.com/webhook/path/to/resource"
    domain1, path1 = handle_webhook_url(test_url_1)
    print(f"URL: {test_url_1}")
    print(f"Domain: {domain1}")
    print(f"Path: {path1}")
    print("-" * 20)

    test_url_2 = "http://localhost:8000/api/v1/data"
    domain2, path2 = handle_webhook_url(test_url_2)
    print(f"URL: {test_url_2}")
    print(f"Domain: {domain2}")
    print(f"Path: {path2}")
    print("-" * 20)

    test_url_3 = "invalid-url"
    domain3, path3 = handle_webhook_url(test_url_3)
    print(f"URL: {test_url_3}")
    print(f"Domain: {domain3}") # Expected: '' or None
    print(f"Path: {path3}")   # Expected: 'invalid-url' or None
    print("-" * 20)

    test_url_4 = "ftp://files.example.com/uploads/report.pdf"
    domain4, path4 = handle_webhook_url(test_url_4)
    print(f"URL: {test_url_4}")
    print(f"Domain: {domain4}")
    print(f"Path: {path4}")
    print("-" * 20)
