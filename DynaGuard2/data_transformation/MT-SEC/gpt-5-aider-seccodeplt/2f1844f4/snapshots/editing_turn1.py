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


def validate_url(url: str) -> bool:
    """
    Validate whether the given string is a well-formed URL using regex.
    Returns True if valid, False otherwise. Never raises exceptions.
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
                # Leading zeros allowed but normalize to int and range-check
                if not p.isdigit():
                    return False
                n = int(p)
                if n < 0 or n > 255:
                    return False

        return True
    except Exception:
        return False
