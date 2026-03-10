from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary with the URL components (scheme, domain, path).
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "domain": parsed_url.netloc,
        "path": parsed_url.path,
    }

if __name__ == '__main__':
    # Example usage:
    test_url = "https://www.example.com/path/to/resource?query=param#fragment"
    components = extract_url_components(test_url)
    print(f"URL: {test_url}")
    print(f"Scheme: {components['scheme']}")
    print(f"Domain: {components['domain']}")
    print(f"Path: {components['path']}")

    test_url_2 = "http://localhost:8080/api/v1/users?id=123"
    components_2 = extract_url_components(test_url_2)
    print(f"\nURL: {test_url_2}")
    print(f"Scheme: {components_2['scheme']}")
    print(f"Domain: {components_2['domain']}")
    print(f"Path: {components_2['path']}")

    test_url_3 = "ftp://user:password@ftp.example.com/files/archive.zip"
    components_3 = extract_url_components(test_url_3)
    print(f"\nURL: {test_url_3}")
    print(f"Scheme: {components_3['scheme']}")
    print(f"Domain: {components_3['domain']}")
    print(f"Path: {components_3['path']}")
