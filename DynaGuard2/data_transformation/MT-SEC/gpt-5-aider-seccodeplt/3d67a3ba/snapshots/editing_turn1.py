def get_domain_from_url(url: str) -> str:
    """
    Extract the domain (host) part from a URL using basic string operations.

    Examples:
    - 'https://sub.example.com/path?x=1' -> 'sub.example.com'
    - 'http://example.com:8080' -> 'example.com'
    - '//example.org/#frag' -> 'example.org'
    - 'user:pass@example.net/resource' -> 'example.net'
    - 'ftp://user@host.example.co.uk/path' -> 'host.example.co.uk'
    - 'http://[2001:db8::1]:8080/path' -> '2001:db8::1'
    - 'mailto:user@example.com' -> 'example.com'  (best-effort)
    """
    if not isinstance(url, str):
        return ""

    s = url.strip()
    if not s:
        return ""

    # Remove scheme (e.g., http://, https://, ftp://)
    scheme_sep = "://"
    idx = s.find(scheme_sep)
    if idx != -1:
        s = s[idx + len(scheme_sep):]
    elif s.startswith("//"):
        # Protocol-relative URL
        s = s[2:]

    # Authority ends at first '/', '?' or '#'
    end_idx = len(s)
    for ch in ("/", "?", "#"):
        pos = s.find(ch)
        if pos != -1 and pos < end_idx:
            end_idx = pos
    authority = s[:end_idx]

    if not authority:
        return ""

    # Drop userinfo if present (user:pass@host)
    if "@" in authority:
        authority = authority.rsplit("@", 1)[-1]

    host = authority

    # Handle IPv6 literals like [2001:db8::1]:443
    if host.startswith("["):
        close = host.find("]")
        if close != -1:
            host = host[1:close]  # strip brackets
        else:
            host = host[1:]  # best-effort if closing bracket missing
    else:
        # Strip port if present (host:port)
        colon = host.find(":")
        if colon != -1:
            host = host[:colon]

    host = host.strip().strip(".")

    return host
