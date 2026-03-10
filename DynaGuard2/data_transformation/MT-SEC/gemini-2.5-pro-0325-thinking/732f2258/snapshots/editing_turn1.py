from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the URL components (scheme, netloc, path, params, query, fragment).
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,  # Domain name
        "path": parsed_url.path,
        "params": parsed_url.params,
        "query": parsed_url.query,
        "fragment": parsed_url.fragment,
    }

if __name__ == '__main__':
    # Example usage:
    test_url = "https://www.example.com/path/to/resource?param1=value1&param2=value2#section1"
    components = extract_url_components(test_url)
    print(f"URL: {test_url}")
    print(f"Components: {components}")

    test_url_2 = "http://localhost:8080/api/users?id=123"
    components_2 = extract_url_components(test_url_2)
    print(f"\nURL: {test_url_2}")
    print(f"Components: {components_2}")

    test_url_3 = "ftp://user:password@host.com/data.txt"
    components_3 = extract_url_components(test_url_3)
    print(f"\nURL: {test_url_3}")
    print(f"Components: {components_3}")
