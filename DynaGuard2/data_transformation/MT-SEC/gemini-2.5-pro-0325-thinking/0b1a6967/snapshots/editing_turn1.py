from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the URL components (scheme, hostname, path, params, query, fragment).
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "hostname": parsed_url.hostname,
        "path": parsed_url.path,
        "params": parsed_url.params,
        "query": parsed_url.query,
        "fragment": parsed_url.fragment,
        "netloc": parsed_url.netloc, # often includes port
        "port": parsed_url.port
    }

if __name__ == '__main__':
    # Example usage:
    test_url = "https://www.example.com:8080/path/to/resource?param1=value1&param2=value2#section1"
    url_parts = extract_url_parts(test_url)
    print(f"URL: {test_url}")
    for key, value in url_parts.items():
        print(f"{key}: {value}")

    print("\nAnother example:")
    test_url_2 = "ftp://user:password@ftp.example.co.uk/files/archive.zip;type=i"
    url_parts_2 = extract_url_parts(test_url_2)
    print(f"URL: {test_url_2}")
    for key, value in url_parts_2.items():
        print(f"{key}: {value}")
