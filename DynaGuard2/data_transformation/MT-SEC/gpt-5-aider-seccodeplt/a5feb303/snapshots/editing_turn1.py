from urllib.parse import urlparse

def extract_domain(url: str) -> str:
    """
    Extract the domain (hostname) from a URL string.

    Uses urllib.parse.urlparse to handle parsing. Returns an empty string if a domain
    cannot be determined.

    Examples:
      - "https://user:pass@example.com:8080/path" -> "example.com"
      - "http://127.0.0.1:5000" -> "127.0.0.1"
      - "example.com/path" -> "example.com"
      - "[2001:db8::1]:8080" -> "2001:db8::1"
    """
    if url is None:
        return ""

    u = url.strip()
    if not u:
        return ""

    # Ensure urlparse can detect the netloc even if scheme is missing
    to_parse = u if "://" in u or u.startswith("//") else f"//{u}"
    parsed = urlparse(to_parse)

    host = parsed.hostname or ""
    # Normalize: remove any trailing dot from FQDNs
    return host.rstrip(".")
