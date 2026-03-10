from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components such as scheme, hostname, and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the scheme, hostname, and path of the URL.
        The values might be empty strings or None if a component is not present
        or if the URL is malformed in a way that urlparse handles gracefully.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "hostname": parsed_url.hostname,
        "path": parsed_url.path,
    }

if __name__ == '__main__':
    # Example Usage
    urls_to_test = [
        "https://www.example.com/path/to/resource?query=param#fragment",
        "http://localhost:8080/api/v1/users",
        "ftp://ftp.example.com/files/archive.zip",
        "//example.com/path",  # Scheme-relative URL
        "/just/a/path",        # Only a path
        "mailto:user@example.com",
        "urn:isbn:0451450523"
    ]

    for url in urls_to_test:
        parts = extract_url_parts(url)
        print(f"URL: {url}")
        print(f"  Scheme: {parts['scheme']}")
        print(f"  Hostname: {parts['hostname']}")
        print(f"  Path: {parts['path']}")
        print("-" * 20)
