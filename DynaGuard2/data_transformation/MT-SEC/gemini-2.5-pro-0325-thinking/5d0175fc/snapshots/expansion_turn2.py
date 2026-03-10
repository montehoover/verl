from urllib.parse import urlparse

def extract_protocol(url_string: str) -> str | None:
    """
    Extracts the protocol from a given URL string.

    Args:
        url_string: The URL string to parse.

    Returns:
        The protocol (e.g., 'http', 'https') if present, otherwise None.
    """
    parsed_url = urlparse(url_string)
    if parsed_url.scheme:
        return parsed_url.scheme
    return None


def parse_url_components(url_string: str) -> dict[str, str | None]:
    """
    Parses a URL string and extracts its protocol, domain, and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary with keys 'protocol', 'domain', and 'path'.
        Values can be None if a component is not present.
    """
    parsed_url = urlparse(url_string)
    return {
        "protocol": parsed_url.scheme if parsed_url.scheme else None,
        "domain": parsed_url.netloc if parsed_url.netloc else None,
        "path": parsed_url.path if parsed_url.path else None,
    }

if __name__ == '__main__':
    # Example Usage
    urls_to_test = [
        "http://www.example.com",
        "https://www.example.com/path?query=value",
        "ftp://ftp.example.com",
        "www.example.com",  # No protocol
        "://www.example.com", # Invalid protocol
        "example.com", # No protocol
        "http:/invalid-url", # Malformed but has scheme
        "", # Empty string
    ]

    for url in urls_to_test:
        protocol = extract_protocol(url)
        print(f"URL: '{url}', Protocol: {protocol}")

    # Test with a more complex case
    url_with_port = "https://localhost:8080/api/data"
    print(f"URL: '{url_with_port}', Protocol: {extract_protocol(url_with_port)}")

    print("\n--- Testing parse_url_components ---")
    urls_for_components_test = [
        "http://www.example.com/some/path?query=string#fragment",
        "https://api.example.co.uk:8080/v1/users?id=123",
        "ftp://ftp.fileserver.com/public/data.zip",
        "www.no-protocol.com/path",
        "/just/a/path",
        "http://domainonly.com",
        "https://domainwithslash.com/",
        "",
        "mailto:user@example.com" # scheme is mailto, no domain or path in http sense
    ]

    for url in urls_for_components_test:
        components = parse_url_components(url)
        print(f"URL: '{url}'")
        print(f"  Protocol: {components['protocol']}")
        print(f"  Domain:   {components['domain']}")
        print(f"  Path:     {components['path']}")
        print("-" * 20)
