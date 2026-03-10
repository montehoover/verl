from urllib.parse import urlparse

def parse_url(url_string):
    """
    Parses a URL string and returns its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A ParseResult object with attributes for scheme, netloc, path,
        params, query, and fragment.
    """
    return urlparse(url_string)

if __name__ == '__main__':
    # Example usage:
    url_to_parse = "http://www.example.com/path/to/page?query=value#fragment"
    parsed_components = parse_url(url_to_parse)
    
    print(f"URL: {url_to_parse}")
    print(f"Scheme: {parsed_components.scheme}")
    print(f"Netloc: {parsed_components.netloc}")
    print(f"Path: {parsed_components.path}")
    print(f"Params: {parsed_components.params}")
    print(f"Query: {parsed_components.query}")
    print(f"Fragment: {parsed_components.fragment}")

    url_to_parse_2 = "https://user:password@subdomain.example.co.uk:8080/another/path?name=test&age=30#section2"
    parsed_components_2 = parse_url(url_to_parse_2)

    print(f"\nURL: {url_to_parse_2}")
    print(f"Scheme: {parsed_components_2.scheme}")
    print(f"Netloc: {parsed_components_2.netloc}")
    print(f"Path: {parsed_components_2.path}")
    print(f"Params: {parsed_components_2.params}")
    print(f"Query: {parsed_components_2.query}")
    print(f"Fragment: {parsed_components_2.fragment}")
    print(f"Hostname: {parsed_components_2.hostname}")
    print(f"Port: {parsed_components_2.port}")
    print(f"Username: {parsed_components_2.username}")
    print(f"Password: {parsed_components_2.password}")
