from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Extracts components like scheme, domain, and path from a URL string.

    Args:
        url_string (str): The URL string to parse.

    Returns:
        dict: A dictionary containing the scheme, domain (netloc), and path
              of the URL. Returns None if the URL cannot be parsed.
    """
    try:
        parsed_url = urlparse(url_string)
        return {
            "scheme": parsed_url.scheme,
            "domain": parsed_url.netloc,
            "path": parsed_url.path,
        }
    except Exception:
        # Handle potential errors during parsing, though urlparse is quite robust
        return None

if __name__ == '__main__':
    # Example usage:
    test_url_1 = "https://www.example.com/path/to/resource?query=param#fragment"
    components_1 = extract_url_components(test_url_1)
    print(f"Components of '{test_url_1}': {components_1}")

    test_url_2 = "ftp://myuser:mypass@localhost:2121/files/archive.zip"
    components_2 = extract_url_components(test_url_2)
    print(f"Components of '{test_url_2}': {components_2}")

    test_url_3 = "invalid-url"
    components_3 = extract_url_components(test_url_3)
    print(f"Components of '{test_url_3}': {components_3}")
