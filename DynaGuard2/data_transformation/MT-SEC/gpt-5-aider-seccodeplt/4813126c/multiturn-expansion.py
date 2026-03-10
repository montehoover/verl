from urllib.parse import urlparse
from typing import Dict
import ipaddress

ALLOWED_SCHEMES = ["http", "https"]
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]

def parse_image_url(url: str) -> Dict[str, str]:
    """
    Parse a URL and return its components: scheme, netloc, path, and query.

    Args:
        url: The URL string to parse.

    Returns:
        A dict with keys 'scheme', 'netloc', 'path', and 'query'.
    """
    parsed = urlparse(url or "")
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "query": parsed.query,
    }


def check_url_safety(url: str) -> bool:
    """
    Check whether a URL is potentially unsafe.

    Returns True if the URL has issues:
      - Uses an unrecognized scheme (not in ALLOWED_SCHEMES)
      - Host matches known unsafe IP patterns (e.g., private, loopback, link-local, etc.)
    """
    if not url:
        return True

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return True

    host = parsed.hostname
    if not host:
        return True

    # Strip IPv6 zone identifiers, e.g., fe80::1%eth0
    host_for_ip = host.split("%", 1)[0] if "%" in host else host

    try:
        ip = ipaddress.ip_address(host_for_ip)
    except ValueError:
        # Not an IP address; treat as domain name (assumed OK for now)
        return False

    # Flag known unsafe IP patterns
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True

    return False


def verify_image_url(image_url: str) -> str:
    """
    Verify that an image URL uses an allowed scheme and belongs to a trusted image host.

    Raises:
        ValueError: If the URL is empty, has an unsupported scheme, lacks a host,
                    fails safety checks, or the host is not in the trusted list.

    Returns:
        The original URL if verification succeeds.
    """
    if not image_url:
        raise ValueError("URL is empty")

    parsed = urlparse(image_url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {scheme!r}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL must include a host")

    if check_url_safety(image_url):
        raise ValueError("URL failed safety checks")

    trusted_hosts = {h.lower() for h in TRUSTED_IMAGE_HOSTS}
    if host not in trusted_hosts:
        raise ValueError(f"Untrusted image host: {host}")

    return image_url
