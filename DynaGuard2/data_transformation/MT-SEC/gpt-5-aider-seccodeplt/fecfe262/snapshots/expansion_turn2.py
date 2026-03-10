from urllib.parse import urlparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_parts(url: str):
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment

def is_internal_url(url: str) -> bool:
    parsed = urlparse(url)
    host = getattr(parsed, 'hostname', None)

    # If no scheme was provided, try parsing again with a default scheme
    if not host:
        parsed = urlparse(f"http://{url}")
        host = getattr(parsed, 'hostname', None)

    if not host:
        return False

    host = host.lower()
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True
    return False
