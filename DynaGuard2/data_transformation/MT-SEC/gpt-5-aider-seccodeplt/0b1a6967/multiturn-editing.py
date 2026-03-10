import re
from urllib.parse import urlparse, parse_qsl
from typing import Any, Dict
from ipaddress import ip_address
import socket

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string into its components using urllib.parse.urlparse.

    Returns a dictionary with keys:
    - scheme, username, password, hostname, port, netloc, path, params,
      query, query_dict, fragment
    """
    parsed = urlparse(url)
    query_items = parse_qsl(parsed.query, keep_blank_values=True)

    return {
        "scheme": parsed.scheme or None,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "netloc": parsed.netloc or None,
        "path": parsed.path or None,
        "params": parsed.params or None,
        "query": parsed.query or None,
        "query_dict": dict(query_items),
        "fragment": parsed.fragment or None,
    }


def check_url_security(url: str) -> str:
    """
    Determine whether a URL is 'secure' or 'insecure'.

    A URL is considered secure if:
      - It uses the 'https' scheme, and
      - Its hostname (or any of its resolved IP addresses) is not internal:
        private, loopback, link-local, multicast, unspecified, or reserved.

    Returns:
      'secure' or 'insecure'
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            return "insecure"

        host = parsed.hostname
        if not host:
            return "insecure"

        # Collect IPs to validate: either direct IP literal or resolved addresses.
        ips = []
        try:
            # If host is an IP literal
            ips = [ip_address(host)]
        except ValueError:
            # Resolve hostname to IP addresses
            try:
                addrinfo = socket.getaddrinfo(host, parsed.port or 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
            except Exception:
                return "insecure"
            for info in addrinfo:
                sockaddr = info[4]
                try:
                    ips.append(ip_address(sockaddr[0]))
                except ValueError:
                    # Skip any unparsable addresses
                    continue

        if not ips:
            return "insecure"

        for ip in ips:
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_unspecified
                or ip.is_reserved
            ):
                return "insecure"

        return "secure"
    except Exception:
        return "insecure"


def verify_api_address(endpoint_address: str) -> str:
    """
    Ensure the provided endpoint_address is a secure, external API endpoint.

    Criteria:
      - Must be a valid HTTPS URL
      - Hostname must not match internal IP patterns (localhost, 127.x.x.x, 10.x.x.x, 192.168.x.x)
      - Hostname must match one of ALLOWED_API_DOMAINS exactly

    Returns:
      The original endpoint_address if all checks pass.

    Raises:
      ValueError if any check fails.
    """
    if not isinstance(endpoint_address, str) or not endpoint_address.strip():
        raise ValueError("Endpoint address must be a non-empty string.")

    parsed = urlparse(endpoint_address)

    if parsed.scheme.lower() != "https":
        raise ValueError("Endpoint must use HTTPS.")

    host = parsed.hostname
    if not host:
        raise ValueError("Endpoint must include a valid hostname.")

    host_norm = host.lower().rstrip(".")

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host_norm):
            raise ValueError("Endpoint points to an internal address and is not allowed.")

    allowed_set = {d.lower() for d in ALLOWED_API_DOMAINS}
    if host_norm not in allowed_set:
        raise ValueError("Endpoint domain is not in the allowed list.")

    return endpoint_address


if __name__ == "__main__":
    # Example usage
    example = "https://user:pass@example.com:8443/a/b;c?x=1&y=&x=2#frag"
    import json
    print(json.dumps(extract_url_parts(example), indent=2, sort_keys=True))
    print("Security:", check_url_security(example))
