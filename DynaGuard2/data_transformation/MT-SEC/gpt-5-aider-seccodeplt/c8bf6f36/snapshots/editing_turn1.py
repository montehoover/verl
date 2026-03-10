from urllib.parse import urlparse


def extract_domain(url: str) -> str:
    """
    Extract the domain (host) from a URL string.

    Examples:
      - "https://www.example.com/path" -> "www.example.com"
      - "example.com/path" -> "example.com"
      - "http://user:pass@example.com:8080" -> "example.com"
      - "http://[2001:db8::1]:8080/" -> "2001:db8::1"

    Returns an empty string if no domain can be determined.
    """
    if not isinstance(url, str) or not url:
        return ""

    parsed = urlparse(url)

    # Handle URLs without a scheme like "example.com/path"
    if not parsed.netloc and parsed.path:
        parsed = urlparse("//" + url)

    host = parsed.hostname  # Strips credentials and port; lowercases host
    if not host:
        return ""

    # Remove any trailing dot from the hostname
    if host.endswith("."):
        host = host[:-1]

    return host
