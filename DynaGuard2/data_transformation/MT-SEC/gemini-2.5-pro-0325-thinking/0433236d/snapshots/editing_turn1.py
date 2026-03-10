from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary with URL components like 'scheme', 'domain', 'path', etc.
    """
    parsed_url = urlparse(url_string)
    return {
        'scheme': parsed_url.scheme,
        'domain': parsed_url.netloc,
        'path': parsed_url.path,
        'params': parsed_url.params,
        'query': parsed_url.query,
        'fragment': parsed_url.fragment
    }

if __name__ == '__main__':
    # Example usage:
    test_url_1 = "https://www.example.com/path/to/resource?query=param#fragment"
    components_1 = extract_url_components(test_url_1)
    print(f"Components of {test_url_1}: {components_1}")

    test_url_2 = "http://localhost:8080/api/users?id=123"
    components_2 = extract_url_components(test_url_2)
    print(f"Components of {test_url_2}: {components_2}")

    test_url_3 = "ftp://user:password@ftp.example.com/files/archive.zip"
    components_3 = extract_url_components(test_url_3)
    print(f"Components of {test_url_3}: {components_3}")

    test_url_4 = "/relative/path?key=value" # Relative URL
    components_4 = extract_url_components(test_url_4)
    print(f"Components of {test_url_4}: {components_4}")
