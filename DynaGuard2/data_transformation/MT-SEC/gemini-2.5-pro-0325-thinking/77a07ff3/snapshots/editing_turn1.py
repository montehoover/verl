from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the scheme, domain, and path of the URL.
        Returns None if the URL cannot be parsed.
    """
    try:
        parsed_url = urlparse(url_string)
        return {
            "scheme": parsed_url.scheme,
            "domain": parsed_url.netloc,
            "path": parsed_url.path,
            "params": parsed_url.params,
            "query": parsed_url.query,
            "fragment": parsed_url.fragment,
        }
    except Exception:
        # Handle potential errors during parsing, though urlparse is quite robust
        return None

if __name__ == '__main__':
    # Example Usage
    test_url_1 = "http://www.example.com/path/to/resource?query=param#fragment"
    components_1 = extract_url_components(test_url_1)
    if components_1:
        print(f"Components of {test_url_1}:")
        for key, value in components_1.items():
            print(f"  {key}: {value}")
    else:
        print(f"Could not parse {test_url_1}")

    print("-" * 20)

    test_url_2 = "https://subdomain.example.co.uk:8080/another/path.html?name=test&value=123"
    components_2 = extract_url_components(test_url_2)
    if components_2:
        print(f"Components of {test_url_2}:")
        for key, value in components_2.items():
            print(f"  {key}: {value}")
    else:
        print(f"Could not parse {test_url_2}")

    print("-" * 20)
    
    test_url_3 = "ftp://user:password@host.com/data"
    components_3 = extract_url_components(test_url_3)
    if components_3:
        print(f"Components of {test_url_3}:")
        for key, value in components_3.items():
            print(f"  {key}: {value}")
    else:
        print(f"Could not parse {test_url_3}")

    print("-" * 20)

    # Example of a simpler URL
    test_url_4 = "example.com/justapath" 
    # urlparse will try to infer scheme if missing, or treat 'example.com' as path if no '//'
    components_4 = extract_url_components(test_url_4)
    if components_4:
        print(f"Components of {test_url_4}:")
        for key, value in components_4.items():
            print(f"  {key}: {value}") # scheme might be empty
    else:
        print(f"Could not parse {test_url_4}")
