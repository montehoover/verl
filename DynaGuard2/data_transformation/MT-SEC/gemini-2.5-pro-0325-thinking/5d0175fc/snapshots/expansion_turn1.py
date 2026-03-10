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
