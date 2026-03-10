import re
import socket
import ipaddress
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


def verify_api_address(endpoint_address: str) -> str:
    """
    Verify that the provided URL points to an approved, secure external API endpoint and
    does not target internal networks (mitigating SSRF risks).

    Args:
        endpoint_address (str): The URL that needs to be verified.

    Returns:
        str: The verified URL if it is allowed.

    Raises:
        ValueError: If the URL is invalid, uses an insecure scheme, contains userinfo,
                    targets an unapproved domain, or targets an internal/special address.
    """
    if not isinstance(endpoint_address, str) or not endpoint_address.strip():
        raise ValueError("endpoint_address must be a non-empty string URL.")

    url = endpoint_address.strip()
    parsed = urlparse(url)

    # Require HTTPS
    if parsed.scheme.lower() != "https":
        raise ValueError("Only HTTPS API endpoints are allowed.")

    # Must include a network location (host)
    if not parsed.netloc:
        raise ValueError("Invalid URL: missing network location (host).")

    # Disallow userinfo to prevent obfuscated hosts like user@host
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Userinfo (username/password) in URL is not allowed.")

    # Extract and normalize hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: missing hostname.")

    host_norm = hostname.rstrip(".").lower()

    # Convert to ASCII/Punycode to prevent homograph tricks
    try:
        host_ascii = host_norm.encode("idna").decode("ascii")
    except Exception:
        raise ValueError("Invalid hostname encoding.")

    # Quick blocklist via regex patterns (localhost, loopback, private IPv4 ranges)
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host_ascii):
            raise ValueError("URL points to an internal address, which is not allowed.")

    # If hostname is an IP literal, ensure it is public and routable
    try:
        ip_obj = ipaddress.ip_address(host_ascii)
        if (
            ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            raise ValueError("URL resolves to an internal or special IP address, which is not allowed.")
    except ValueError:
        # Not an IP literal; continue checks
        pass

    # Enforce allowed domains (exact match)
    allowed = {d.lower() for d in ALLOWED_API_DOMAINS}
    if host_ascii not in allowed:
        raise ValueError("Domain is not in the allowed API domains list.")

    # DNS resolution check: ensure resolved IPs are not internal/special (mitigate DNS rebinding)
    try:
        # Use default HTTPS port if not specified for resolution purposes
        port_for_resolution = parsed.port or 443
        addrinfo_list = socket.getaddrinfo(host_ascii, port_for_resolution, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ValueError("DNS resolution failed for the API hostname.")

    resolved_ips = set()
    for family, _, _, _, sockaddr in addrinfo_list:
        if family == socket.AF_INET:
            ip_str = sockaddr[0]
        elif family == socket.AF_INET6:
            ip_str = sockaddr[0]
        else:
            continue
        resolved_ips.add(ip_str)

    if not resolved_ips:
        raise ValueError("Unable to resolve the API hostname to any IP address.")

    for ip_str in resolved_ips:
        ip_obj = ipaddress.ip_address(ip_str)
        if (
            ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            raise ValueError("Resolved IP address is not allowed.")

    # All checks passed; return the original URL
    return url
