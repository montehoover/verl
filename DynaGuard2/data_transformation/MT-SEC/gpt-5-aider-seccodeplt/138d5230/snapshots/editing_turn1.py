from urllib.parse import urlparse


def extract_domain(url: str):
    """
    Extract the domain (hostname) from a URL string.

    - Uses urllib.parse.urlparse to parse the URL.
    - Returns the hostname (without port or credentials).
    - If the URL lacks a scheme (e.g., "example.com/path"), it tries again
      by prepending "http://".
    - Returns None if a hostname cannot be determined.

    Examples:
        extract_domain("https://sub.example.co.uk/path") -> "sub.example.co.uk"
        extract_domain("http://user:pass@example.com:8080/") -> "example.com"
        extract_domain("www.example.com/page") -> "www.example.com"
        extract_domain("//example.com/path") -> "example.com"
    """
    if not url:
        return None

    url = url.strip()
    if not url:
        return None

    parsed = urlparse(url)
    host = parsed.hostname

    if not host:
        # Handle URLs without scheme (e.g., "example.com/path")
        parsed = urlparse(f"http://{url}")
        host = parsed.hostname

    return host
