from urllib.parse import urlparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def parse_url(url):
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path

def is_internal_url(url):
    parsed = urlparse(url)
    host = parsed.hostname

    if not host:
        # Attempt to parse schemeless URLs like "localhost" or "192.168.1.1/path"
        parsed_no_scheme = urlparse("//" + url)
        host = parsed_no_scheme.hostname

    if not host:
        return False

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True
    return False
