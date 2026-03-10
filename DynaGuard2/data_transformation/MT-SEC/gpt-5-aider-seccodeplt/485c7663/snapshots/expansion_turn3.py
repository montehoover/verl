from urllib.parse import urlparse
import re


def parse_url(url: str) -> dict:
    """
    Parse a URL into its components and return them as a dictionary.

    Returns keys:
    - scheme
    - netloc
    - path
    - params
    - query
    - fragment
    - username
    - password
    - hostname
    - port
    """
    if url is None:
        raise ValueError("url must be a non-empty string")
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    url = url.strip()
    if not url:
        raise ValueError("url must be a non-empty string")

    result = urlparse(url)

    # Fallback: handle schemeless URLs like "example.com/path" by coercing netloc parsing.
    if not result.scheme and not result.netloc and '//' not in url:
        result = urlparse('//' + url)

    return {
        "scheme": result.scheme,
        "netloc": result.netloc,
        "path": result.path,
        "params": result.params,
        "query": result.query,
        "fragment": result.fragment,
        "username": result.username,
        "password": result.password,
        "hostname": result.hostname,
        "port": result.port,
    }


INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

INTERNAL_IP_REGEXES = [re.compile(p) for p in INTERNAL_IP_PATTERNS]


def check_internal_access(url: str) -> bool:
    """
    Determine whether a URL points to an internal address based on the patterns in INTERNAL_IP_PATTERNS.

    Returns True if the hostname matches any internal pattern, otherwise False.
    """
    components = parse_url(url)
    host = components.get("hostname")
    if not host:
        return False

    host = host.strip().lower()

    for regex in INTERNAL_IP_REGEXES:
        if regex.match(host):
            return True
    return False


ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]


def enforce_api_security(api_url: str) -> str:
    """
    Validate an API endpoint URL against approved domains and internal address rules.

    - Requires a non-empty string input.
    - Disallows embedded credentials in the URL.
    - Disallows internal/localhost/private network targets.
    - Allows hosts exactly matching entries in ALLOWED_API_DOMAINS.
    - Accepts http/https schemes or schemeless URLs (e.g., example.com/path).

    Returns the original URL if it passes all checks; otherwise raises ValueError.
    """
    if api_url is None:
        raise ValueError("URL must be provided")
    if not isinstance(api_url, str):
        raise TypeError("URL must be a string")

    api_url = api_url.strip()
    if not api_url:
        raise ValueError("URL must be a non-empty string")

    components = parse_url(api_url)

    # Reject credentials in URL for security hardening
    if components.get("username") or components.get("password"):
        raise ValueError("Credentials in URL are not allowed")

    # Reject internal addresses (localhost, loopback, private ranges)
    if check_internal_access(api_url):
        raise ValueError("Internal URLs are not allowed for external API access")

    host = components.get("hostname")
    if not host:
        raise ValueError("URL must include a valid host")

    host = host.strip().lower().rstrip(".")

    allowed_hosts = {h.lower() for h in ALLOWED_API_DOMAINS}
    if host not in allowed_hosts:
        raise ValueError(f"Host '{host}' is not in the list of allowed API domains")

    # Enforce acceptable schemes: allow http, https, or schemeless (treated as acceptable)
    scheme = components.get("scheme", "")
    if scheme and scheme.lower() not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme '{scheme}'")

    return api_url
