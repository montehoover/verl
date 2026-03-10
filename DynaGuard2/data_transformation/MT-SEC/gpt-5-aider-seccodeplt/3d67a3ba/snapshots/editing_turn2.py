import re
from typing import Tuple


def extract_url_components(url: str) -> Tuple[str, str]:
    """
    Extract the protocol (scheme) and domain (host) from a URL using regular expressions.

    Returns a tuple: (protocol, domain). If not extractable, returns ("", "").

    Examples:
    - 'https://sub.example.com/path?x=1' -> ('https', 'sub.example.com')
    - 'http://example.com:8080' -> ('http', 'example.com')
    - '//example.org/#frag' -> ('', 'example.org')
    - 'ftp://user@host.example.co.uk/path' -> ('ftp', 'host.example.co.uk')
    - 'http://[2001:db8::1]:8080/path' -> ('http', '2001:db8::1')
    - 'mailto:user@example.com' -> ('mailto', 'example.com')
    - 'example.net/path' -> ('', 'example.net')
    """
    if not isinstance(url, str):
        return ("", "")

    s = url.strip()
    if not s:
        return ("", "")

    # Special-case mailto: scheme (extract domain after '@')
    m = re.match(r'^(?P<scheme>mailto):[^@/\s]+@(?P<host>[^/?#\s]+)', s, flags=re.IGNORECASE)
    if m:
        scheme = m.group('scheme').lower()
        host = m.group('host').strip().strip('.')
        return (scheme, host)

    # Standard URLs with scheme:// or protocol-relative //
    m = re.match(
        r'^(?:(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*)://|//)'
        r'(?:(?:[^@/?#]*?)@)?'            # optional userinfo
        r'(?P<host>\[[^\]]+\]|[^:/?#\s]+)',  # host (IPv6 in [], or reg-name)
        s
    )
    if m:
        scheme = (m.group('scheme') or "").lower()
        host = m.group('host')
        if host.startswith('[') and host.endswith(']'):
            host = host[1:-1]  # strip brackets from IPv6
        host = host.strip().strip('.')
        return (scheme, host)

    # Userinfo without scheme (e.g., user:pass@host.tld/path)
    m = re.match(r'^(?:[^@/?#]+@)(?P<host>\[[^\]]+\]|[^:/?#\s]+)', s)
    if m:
        host = m.group('host')
        if host.startswith('[') and host.endswith(']'):
            host = host[1:-1]
        host = host.strip().strip('.')
        return ("", host)

    # Bare host without scheme/userinfo, like 'example.com' or '[::1]'
    m = re.match(r'^(?P<host>\[[^\]]+\]|[^:/?#\s]+)', s)
    if m:
        host = m.group('host')
        if host.startswith('[') and host.endswith(']'):
            host = host[1:-1]
        host = host.strip().strip('.')
        return ("", host)

    return ("", "")


def get_domain_from_url(url: str) -> str:
    """
    Backward-compatible helper that returns only the domain.
    """
    _, domain = extract_url_components(url)
    return domain
