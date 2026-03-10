import re
import ipaddress
from urllib.parse import urlsplit, urlunsplit, urljoin, quote
from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


_ALLOWED_SCHEMES = {"http", "https", "ws", "wss", "ftp", "ftps"}


def _is_valid_hostname(hostname: str) -> bool:
    # Accept IP addresses (IPv4/IPv6)
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    # Validate IDNA-encodable hostnames
    try:
        ascii_host = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if len(ascii_host) > 253:
        return False

    # Allow trailing dot in hostname representation for FQDNs
    if ascii_host.endswith("."):
        ascii_host = ascii_host[:-1]

    labels = ascii_host.split(".")
    label_re = re.compile(r"^[A-Za-z0-9-]{1,63}$")

    for label in labels:
        if not label:
            return False
        if not label_re.match(label):
            return False
        if label[0] == "-" or label[-1] == "-":
            return False

    return True


def _normalize_hostname(hostname: str | None) -> str | None:
    if hostname is None:
        return None
    try:
        # If IP, return canonical string form
        ip = ipaddress.ip_address(hostname)
        return ip.exploded if isinstance(ip, ipaddress.IPv6Address) else str(ip)
    except ValueError:
        pass
    try:
        ascii_host = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    # strip trailing dot, lowercase
    if ascii_host.endswith("."):
        ascii_host = ascii_host[:-1]
    return ascii_host.lower()


def _host_matches_or_subdomain(host: str, domain: str) -> bool:
    """
    True if host equals domain or is a subdomain of domain.
    IP addresses must match exactly.
    """
    nhost = _normalize_hostname(host)
    ndomain = _normalize_hostname(domain)
    if nhost is None or ndomain is None:
        return False
    # If either is IP, only exact match is allowed
    is_host_ip = True
    try:
        ipaddress.ip_address(nhost)
    except ValueError:
        is_host_ip = False
    is_domain_ip = True
    try:
        ipaddress.ip_address(ndomain)
    except ValueError:
        is_domain_ip = False

    if is_host_ip or is_domain_ip:
        return nhost == ndomain

    return nhost == ndomain or nhost.endswith("." + ndomain)


def validate_url(url: str) -> bool:
    """
    Validate that the given URL is well-formed.
    Returns True if valid; otherwise raises ValueError.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    candidate = url.strip()
    if not candidate:
        raise ValueError("URL is empty")

    if re.search(r"\s", candidate):
        raise ValueError("URL must not contain whitespace characters")

    parsed = urlsplit(candidate)

    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http, https)")
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    if not parsed.netloc:
        raise ValueError("URL must include a network location (host)")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    if not _is_valid_hostname(hostname):
        raise ValueError("Invalid hostname in URL")

    # Validate port (if present)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid port in URL")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Port out of valid range (1-65535)")

    return True


def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenate a URL path to a base URL and return the full URL.

    Rules:
    - base_url must be a valid URL (validated via validate_url).
    - path must be a string path (no scheme or netloc). It may include query/fragment.
    - If path is empty or whitespace, base_url is returned unchanged.
    - If path is absolute (starts with '/'), it is resolved from the domain root.
    - Query/fragment in path override those in base_url; otherwise base URL's are preserved.
    - Path components are percent-encoded as needed (without double-encoding).
    """
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a string")
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    base = base_url.strip()
    segment = path.strip()

    # Validate base URL
    validate_url(base)

    if segment == "":
        return base

    # Disallow full URLs in the path argument
    path_parts = urlsplit(segment)
    if path_parts.scheme or path_parts.netloc:
        raise ValueError("path must be a URL path, not a full URL")

    # Prepare base URL with trailing slash and without query/fragment for joining
    base_parts = urlsplit(base)
    base_path = base_parts.path if base_parts.path else "/"
    if not base_path.endswith("/"):
        base_path = base_path + "/"
    base_for_join = urlunsplit((base_parts.scheme, base_parts.netloc, base_path, "", ""))

    # Percent-encode the path portion appropriately (preserve '/' and existing encodings)
    encoded_path = quote(path_parts.path, safe="/%:@!$&'()*+,;=~.-")

    joined = urljoin(base_for_join, encoded_path)
    joined_parts = urlsplit(joined)

    # Preserve base query/fragment unless overridden by the provided path
    final_query = path_parts.query if path_parts.query else base_parts.query
    final_fragment = path_parts.fragment if path_parts.fragment else base_parts.fragment

    full_url = urlunsplit(
        (joined_parts.scheme, joined_parts.netloc, joined_parts.path, final_query, final_fragment)
    )

    # Final validation
    validate_url(full_url)

    return full_url


def generate_redirect_link(app_base_url: str, redirect_param: str) -> str:
    """
    Generate a safe redirect URL after login.

    - If redirect_param is a full URL (absolute or scheme-relative), it must point to:
        * the same base domain as app_base_url (same host or its subdomain), or
        * a host within ALLOWED_DOMAINS (exact or subdomain).
      Otherwise, raise ValueError.

    - If redirect_param is a relative path, it will be safely joined to app_base_url.

    Returns the final absolute redirect URL as a string.
    """
    if not isinstance(app_base_url, str):
        raise ValueError("app_base_url must be a string")
    if not isinstance(redirect_param, str):
        raise ValueError("redirect_param must be a string")

    base = app_base_url.strip()
    target = redirect_param.strip()

    # Validate base URL
    validate_url(base)
    base_parts = urlsplit(base)
    if not base_parts.hostname:
        raise ValueError("Base URL must include a hostname")
    base_host_norm = _normalize_hostname(base_parts.hostname)
    if base_host_norm is None:
        raise ValueError("Base URL has invalid hostname")

    if target == "" or target == "/":
        # Default to base URL
        return base

    # Parse the target
    t_parts = urlparse.urlsplit(target)

    # If the target includes a scheme or netloc, it's an absolute (or scheme-relative) URL
    if t_parts.scheme or t_parts.netloc:
        # For scheme-relative URLs (e.g., //host/path), adopt the base scheme
        scheme = t_parts.scheme or base_parts.scheme
        absolute = urlparse.urlunsplit(
            (scheme, t_parts.netloc, t_parts.path, t_parts.query, t_parts.fragment)
        )

        # Validate URL syntax and scheme
        validate_url(absolute)

        # Domain checks
        host = urlparse.urlsplit(absolute).hostname
        if not host:
            raise ValueError("Redirect URL must include a hostname")
        host_norm = _normalize_hostname(host)
        if host_norm is None:
            raise ValueError("Redirect URL has invalid hostname")

        # Is within base domain?
        within_base = _host_matches_or_subdomain(host_norm, base_host_norm)

        # Is within allowed external domains?
        allowed_external = any(
            _host_matches_or_subdomain(host_norm, allowed) for allowed in ALLOWED_DOMAINS
        )

        if not (within_base or allowed_external):
            raise ValueError("Disallowed redirect domain")

        return absolute

    # Relative path: join safely to base URL
    full_url = concatenate_url_path(base, target)

    # Safety: ensure resulting host is still within base domain (relative paths should be)
    host = urlsplit(full_url).hostname
    if not host or not _host_matches_or_subdomain(host, base_host_norm):
        raise ValueError("Disallowed redirect path")

    return full_url
