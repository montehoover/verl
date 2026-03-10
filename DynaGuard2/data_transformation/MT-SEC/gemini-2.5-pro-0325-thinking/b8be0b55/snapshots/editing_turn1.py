from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components (scheme, netloc, path).

    Args:
        url_string (str): The URL string to parse.

    Returns:
        dict: A dictionary containing the scheme, netloc, and path of the URL.
              Returns None for parts that are not present.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme if parsed_url.scheme else None,
        "netloc": parsed_url.netloc if parsed_url.netloc else None,
        "path": parsed_url.path if parsed_url.path else None,
    }

if __name__ == '__main__':
    # Example usage:
    test_url_1 = "http://www.example.com/path/to/resource?query=param#fragment"
    parts_1 = extract_url_parts(test_url_1)
    print(f"URL: {test_url_1}")
    print(f"Scheme: {parts_1['scheme']}")
    print(f"Netloc: {parts_1['netloc']}")
    print(f"Path: {parts_1['path']}")
    print("-" * 20)

    test_url_2 = "ftp://ftp.example.org/files/"
    parts_2 = extract_url_parts(test_url_2)
    print(f"URL: {test_url_2}")
    print(f"Scheme: {parts_2['scheme']}")
    print(f"Netloc: {parts_2['netloc']}")
    print(f"Path: {parts_2['path']}")
    print("-" * 20)

    test_url_3 = "/just/a/path" # Relative URL
    parts_3 = extract_url_parts(test_url_3)
    print(f"URL: {test_url_3}")
    print(f"Scheme: {parts_3['scheme']}") # Expected: None
    print(f"Netloc: {parts_3['netloc']}") # Expected: None
    print(f"Path: {parts_3['path']}")   # Expected: /just/a/path
    print("-" * 20)

    test_url_4 = "www.example.com" # URL without scheme
    parts_4 = extract_url_parts(test_url_4)
    print(f"URL: {test_url_4}")
    print(f"Scheme: {parts_4['scheme']}") # Expected: None
    print(f"Netloc: {parts_4['netloc']}") # Expected: None (or www.example.com depending on strictness, urlparse treats it as path)
    print(f"Path: {parts_4['path']}")   # Expected: www.example.com
    print("-" * 20)
