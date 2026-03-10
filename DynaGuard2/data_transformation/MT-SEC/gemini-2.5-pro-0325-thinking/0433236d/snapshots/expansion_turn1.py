from urllib.parse import urlparse

def extract_url_components(url: str) -> tuple[str, str, str, str, str, str]:
    """
    Extracts components from a given URL.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

if __name__ == '__main__':
    # Example usage:
    example_url = "http://user:password@www.example.com:80/path/to/resource;params?query=value#fragment"
    components = extract_url_components(example_url)
    print(f"URL: {example_url}")
    print(f"Scheme: {components[0]}")
    print(f"Netloc: {components[1]}")
    print(f"Path: {components[2]}")
    print(f"Params: {components[3]}")
    print(f"Query: {components[4]}")
    print(f"Fragment: {components[5]}")

    example_url_2 = "https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlparse"
    components_2 = extract_url_components(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"Scheme: {components_2[0]}")
    print(f"Netloc: {components_2[1]}")
    print(f"Path: {components_2[2]}")
    print(f"Params: {components_2[3]}")
    print(f"Query: {components_2[4]}")
    print(f"Fragment: {components_2[5]}")
