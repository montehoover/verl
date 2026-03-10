from urllib.parse import urlparse

def parse_and_validate_url(url_string: str) -> dict:
    """
    Parses a URL string and validates it.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the URL's components (scheme, netloc, path,
        params, query, fragment).

    Raises:
        ValueError: If the URL is invalid (e.g., missing scheme or domain).
    """
    if not isinstance(url_string, str):
        raise TypeError("URL must be a string.")

    parsed_url = urlparse(url_string)

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: '{url_string}'. URL must have a scheme and domain.")

    return {
        "scheme": parsed_url.scheme,
        "domain": parsed_url.netloc,
        "path": parsed_url.path,
        "params": parsed_url.params,
        "query": parsed_url.query,
        "fragment": parsed_url.fragment,
    }

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path?query=123#fragment",
        "https://sub.example.co.uk:8080/another/path.html",
        "ftp://user:pass@ftp.example.com/dir/file.txt",
        "invalid-url",
        "www.missing-scheme.com",
        "http://", # Missing domain
        None, # Invalid type
        12345 # Invalid type
    ]

    for url in test_urls:
        print(f"Parsing URL: {url}")
        try:
            components = parse_and_validate_url(url)
            print(f"  Components: {components}")
        except (ValueError, TypeError) as e:
            print(f"  Error: {e}")
        print("-" * 20)
