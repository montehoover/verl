import re

# Compiled regular expression for validating URLs
_URL_RE = re.compile(
    r"""
    ^(?:(?:https?|ftp)://)                                   # scheme
    (?:[^\s/?#@]+(?::[^\s/?#@]*)?@)?                         # optional user:pass@
    (?:
        (?P<localhost>localhost)                             # localhost
        |
        (?P<ipv6>\[[0-9A-Fa-f:.]+\])                         # IPv6 in brackets
        |
        (?P<domain>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63})  # domain
        |
        (?P<ipv4>\d{1,3}(?:\.\d{1,3}){3})                    # IPv4
    )
    (?::(?P<port>\d{1,5}))?                                  # optional port
    (?:/[^\s?#]*)*                                           # path
    (?:\?[^\s#]*)?                                           # query
    (?:#[^\s]*)?                                             # fragment
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_domain(url: str):
    """
    Extract the domain/host part from a well-formed URL using regex.
    Returns the domain/host string if valid, otherwise returns False.
    Never raises exceptions.
    """
    try:
        if not isinstance(url, str):
            return False

        s = url.strip()
        if not s:
            return False

        match = _URL_RE.match(s)
        if not match:
            return False

        # Validate port range if present (1-65535)
        port = match.group("port")
        if port is not None:
            port_num = int(port)
            if port_num < 1 or port_num > 65535:
                return False

        # If matched IPv4, ensure each octet is 0-255
        ipv4 = match.group("ipv4")
        if ipv4:
            parts = ipv4.split(".")
            for p in parts:
                if not p.isdigit():
                    return False
                n = int(p)
                if n < 0 or n > 255:
                    return False
            return ipv4

        # If matched IPv6, strip surrounding brackets
        ipv6 = match.group("ipv6")
        if ipv6:
            # ipv6 is like "[2001:db8::1]"
            return ipv6[1:-1]

        # If matched localhost
        localhost = match.group("localhost")
        if localhost:
            return localhost

        # Domain name
        domain = match.group("domain")
        if domain:
            return domain

        return False
    except Exception:
        return False


def validate_url(url: str) -> bool:
    """
    Validate whether the given string is a well-formed URL using regex.
    Returns True if valid, False otherwise. Never raises exceptions.
    """
    try:
        return bool(extract_domain(url))
    except Exception:
        return False
