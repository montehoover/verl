import re
from urllib.parse import urlparse
import ipaddress

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]

def extract_url_parts(url: str) -> dict:
    """
    Extract components of a URL string into a dictionary.

    Returns keys: scheme, netloc, path, params, query, fragment, username, password, hostname, port.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
    }

def is_valid_url(url: str) -> bool:
    """
    Validate a URL based on:
    - Scheme must be 'http' or 'https'
    - Hostname must not be localhost or a loopback IP (e.g., 127.0.0.1, ::1)

    Returns:
        True if valid, False otherwise.
    """
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False

        if not parsed.netloc:
            return False

        # Accessing parsed.port triggers validation of port number; ValueError if invalid.
        try:
            _ = parsed.port
        except ValueError:
            return False

        host = parsed.hostname
        if not host:
            return False

        host_l = host.lower()

        # Disallow localhost hostnames (including subdomains of .localhost)
        if host_l == "localhost" or host_l.endswith(".localhost"):
            return False

        # If it's an IP address, ensure it's not loopback.
        try:
            ip = ipaddress.ip_address(host_l)
            if ip.is_loopback:
                return False
        except ValueError:
            # Not an IP literal; allow non-localhost hostnames.
            pass

        return True
    except Exception:
        return False

def check_rss_source(source_url: str) -> str:
    """
    Verify an RSS feed URL is from a trusted provider and not accessing internal networks.

    Rules:
    - Scheme must be one of ALLOWED_SCHEMES.
    - Hostname must not match INTERNAL_IP_PATTERNS.
    - Hostname must be exactly a trusted provider or a subdomain of one in TRUSTED_RSS_PROVIDERS.

    Returns:
        The original source_url if valid.

    Raises:
        ValueError: If the URL is invalid or untrusted.
    """
    if not isinstance(source_url, str):
        raise ValueError("source_url must be a string")

    parsed = urlparse(source_url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme must be http or https")

    # Validate port if present (raises ValueError on invalid ports)
    try:
        _ = parsed.port
    except ValueError:
        raise ValueError("Invalid port in URL")

    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a hostname")

    host_l = host.lower()

    # Block obvious internal targets
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host_l):
            raise ValueError("URL targets an internal or loopback address")

    # Enforce trusted providers (exact match or subdomain)
    is_trusted = any(
        host_l == provider or host_l.endswith("." + provider)
        for provider in TRUSTED_RSS_PROVIDERS
    )
    if not is_trusted:
        raise ValueError("Untrusted RSS provider")

    return source_url
