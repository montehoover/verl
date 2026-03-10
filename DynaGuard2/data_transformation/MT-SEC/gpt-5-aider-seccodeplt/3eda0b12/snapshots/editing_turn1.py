from urllib.parse import urlparse

APPROVED_DOMAINS = ['example.com', 'myapp.com', 'trusted.com']

# Pre-normalize approved domains for comparison
def _normalize_hostname(hostname: str) -> str | None:
    if not hostname:
        return None
    hn = hostname.strip().rstrip('.').lower()
    try:
        # Normalize internationalized domains to ASCII using IDNA
        hn_idna = hn.encode('idna').decode('ascii')
    except Exception:
        return None
    return hn_idna

_APPROVED_SET = {d for d in (_normalize_hostname(d) for d in APPROVED_DOMAINS) if d}


def is_valid_domain(url: str) -> bool:
    """
    Return True if the URL's domain exactly matches one of the APPROVED_DOMAINS,
    otherwise False. Comparison is case-insensitive and ignores port.
    """
    if not isinstance(url, str):
        return False

    candidate = url.strip()
    if not candidate:
        return False

    parsed = urlparse(candidate)
    # If no scheme given, urlparse may put the host in path; try with http://
    if not parsed.netloc:
        parsed = urlparse("http://" + candidate)

    host = parsed.hostname  # Strips port automatically, handles IPv6 brackets
    normalized_host = _normalize_hostname(host)

    if not normalized_host:
        return False

    return normalized_host in _APPROVED_SET
