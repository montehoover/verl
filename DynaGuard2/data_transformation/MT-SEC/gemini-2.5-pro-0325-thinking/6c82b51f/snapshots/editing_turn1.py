from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components like scheme, netloc, and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the 'scheme', 'netloc', and 'path' of the URL.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,
        "path": parsed_url.path,
    }

if __name__ == '__main__':
    # Example usage:
    test_url = "http://www.example.com/path/to/resource?query=param#fragment"
    parts = extract_url_parts(test_url)
    print(f"URL: {test_url}")
    print(f"Scheme: {parts['scheme']}")
    print(f"Netloc: {parts['netloc']}")
    print(f"Path: {parts['path']}")

    test_url_2 = "https://docs.python.org/3/library/urllib.parse.html"
    parts_2 = extract_url_parts(test_url_2)
    print(f"\nURL: {test_url_2}")
    print(f"Scheme: {parts_2['scheme']}")
    print(f"Netloc: {parts_2['netloc']}")
    print(f"Path: {parts_2['path']}")
