from urllib.parse import urlparse
from typing import Any, Dict, Optional


def extract_url_components(url: str) -> Dict[str, Any]:
    """
    Extract components from a URL string.

    Returns a dictionary with at least:
      - scheme: URL scheme (e.g., 'http', 'https') or None
      - domain: Hostname without port (e.g., 'example.com') or None
      - path: Path portion of the URL (defaults to '/' if empty)
    Additional fields provided:
      - username: Username from the authority section, if any
      - password: Password from the authority section, if any
      - port: Port number as int, if any
      - query: Query string without the leading '?', if any
      - fragment: Fragment without the leading '#', if any

    The parser attempts to handle schemeless URLs by prepending '//' so that
    the netloc/hostname can still be extracted (e.g., 'example.com/path').

    :param url: The URL string to parse.
    :return: A dictionary of extracted components.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    raw = url.strip()

    parsed = urlparse(raw)
    # If there's no scheme and no netloc, try parsing as a schemeless URL.
    if not parsed.scheme and not parsed.netloc and "://" not in raw:
        parsed = urlparse("//" + raw)

    # Build result dictionary
    result: Dict[str, Optional[Any]] = {
        "scheme": parsed.scheme or None,
        "domain": parsed.hostname or None,  # hostname excludes port and credentials
        "path": parsed.path or "/",
        "username": parsed.username,
        "password": parsed.password,
        "port": parsed.port,
        "query": parsed.query or None,
        "fragment": parsed.fragment or None,
    }

    return result


if __name__ == "__main__":
    # Simple manual test examples
    examples = [
        "https://user:pass@example.com:8443/some/path?x=1#section",
        "http://example.com",
        "example.com/path/to/resource?foo=bar#frag",
        "ftp://[2001:db8::1]:21/dir/file",
        "localhost:3000",
        "https://sub.domain.example.org/",
        "   https://example.com   ",
    ]

    for u in examples:
        print(u, "->", extract_url_components(u))
