import re
import ipaddress
from urllib.parse import urlsplit, urlunsplit, urlencode, quote, parse_qsl
from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


__all__ = ["validate_url", "concatenate_url_path", "construct_redirect_link"]


def _is_valid_domain(host: str) -> bool:
    """
    Validate a domain name according to common DNS rules:
    - Total length <= 253 (in ASCII/IDNA)
    - Each label 1..63 chars, starts/ends with alphanumeric, internal hyphens allowed
    - At least two labels (e.g., example.com)
    - TLD not all-numeric
    """
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if ascii_host.endswith("."):
        ascii_host = ascii_host[:-1]

    if not ascii_host or len(ascii_host) > 253:
        return False

    labels = ascii_host.split(".")
    if len(labels) < 2:
        return False

    label_re = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
    for label in labels:
        if not label or len(label) > 63 or not label_re.fullmatch(label):
            return False

    # TLD cannot be all numeric
    if labels[-1].isdigit():
        return False

    return True


def validate_url(url: str) -> bool:
    """
    Validate a URL intended for web use.
    - Requires http or https scheme
    - Requires a valid host (domain, IP, or 'localhost')
    - Optional port must be 1..65535
    Returns True if valid; raises ValueError if invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty")

    parts = urlsplit(url)

    if parts.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")

    if not parts.netloc:
        raise ValueError("URL must include a host")

    host = parts.hostname
    if not host:
        raise ValueError("URL host is invalid")

    # Validate port if present
    try:
        port = parts.port
    except ValueError:
        raise ValueError("URL port is invalid or out of range")

    if port is not None and not (1 <= port <= 65535):
        raise ValueError("URL port out of range (1-65535)")

    host_norm = host.strip().lower().rstrip(".")

    # Allow localhost
    if host_norm == "localhost":
        return True

    # IP address (IPv4/IPv6)
    try:
        ipaddress.ip_address(host_norm)
        return True
    except ValueError:
        pass

    # Domain name
    if not _is_valid_domain(host_norm):
        raise ValueError("URL host is not a valid domain or IP address")

    return True


def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenate a base URL with a relative path, handling URL encoding.

    - base_url must be a valid http/https URL.
    - path must be a relative URL path (may include query and fragment).
    - Existing encoding in base_url is preserved; path/query/fragment from `path` are percent-encoded.
    - If `path` starts with '/', it is treated as an absolute path from the host root.
    - If `path` is empty or whitespace, the base_url is returned unchanged.
    """
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a string")
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    base_url = base_url.strip()
    if not base_url:
        raise ValueError("base_url cannot be empty")

    # Validate base URL
    validate_url(base_url)

    if path.strip() == "":
        return base_url

    base = urlsplit(base_url)
    p = urlsplit(path)

    # Disallow full URLs in `path` to avoid surprising overrides of base_url
    if p.scheme or p.netloc:
        raise ValueError("path must be a relative URL path, not a full URL")

    # Determine the new path
    base_path = base.path or "/"
    if p.path:
        if p.path.startswith("/"):
            # Absolute from root
            joined_path = "/" + p.path.lstrip("/")
        else:
            # Append to base path as directory
            if not base_path.endswith("/"):
                base_path = base_path + "/"
            joined_path = base_path + p.path
    else:
        # No new path component; keep base path
        joined_path = base_path

    # Percent-encode the resulting path; preserve already-encoded sequences with '%'
    encoded_path = quote(joined_path, safe="/:@-._~!$&'()*+,;=%")

    # Query handling: if `path` provides a query, use it (encoded); otherwise keep base query
    if p.query:
        query_params = parse_qsl(p.query, keep_blank_values=True)
        new_query = urlencode(query_params, doseq=True, quote_via=quote)
    else:
        new_query = base.query

    # Fragment handling: if `path` provides a fragment, use it (encoded); otherwise keep base fragment
    if p.fragment:
        new_fragment = quote(p.fragment, safe="/:@-._~!$&'()*+,;=%")
    else:
        new_fragment = base.fragment

    return urlunsplit((base.scheme, base.netloc, encoded_path, new_query, new_fragment))


def _normalize_host(host: str) -> str:
    if not host:
        return ""
    host = host.strip().lower().rstrip(".")
    try:
        return host.encode("idna").decode("ascii")
    except Exception:
        return host


def _host_is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False


def _host_matches_domain(host_norm: str, domain_norm: str) -> bool:
    """
    True if host is exactly the domain or a subdomain of it.
    """
    if not host_norm or not domain_norm:
        return False
    if host_norm == domain_norm:
        return True
    return host_norm.endswith("." + domain_norm)


def construct_redirect_link(domain_base_url: str, next_redirect_param: str) -> str:
    """
    Construct a safe redirect URL after login.

    - domain_base_url: The base domain URL (http/https), e.g., https://example.com
    - next_redirect_param: A relative path to append to the base, or an absolute URL.
    - The resulting redirect must stay within the base domain (including subdomains),
      or be on an allowed external domain listed in ALLOWED_DOMAINS.
    - Raises ValueError if the redirect target is not allowed.
    """
    if not isinstance(domain_base_url, str):
        raise ValueError("domain_base_url must be a string")

    base_url = domain_base_url.strip()
    if not base_url:
        raise ValueError("domain_base_url cannot be empty")

    # Validate base URL
    validate_url(base_url)
    base_parts = urlparse.urlsplit(base_url)
    base_host_norm = _normalize_host(base_parts.hostname or "")

    if next_redirect_param is None:
        return base_url
    if not isinstance(next_redirect_param, str):
        raise ValueError("next_redirect_param must be a string")

    next_param = next_redirect_param.strip()
    if next_param == "":
        return base_url

    p = urlparse.urlsplit(next_param)

    # Disallow protocol-relative URLs like //evil.com/path
    if not p.scheme and p.netloc:
        raise ValueError("Protocol-relative redirect URLs are not allowed")

    # Build candidate URL
    if p.scheme or p.netloc:
        # Absolute URL: normalize path/query/fragment encoding
        # Validate scheme/host etc.
        validate_url(next_param)
        # Normalize encoding
        encoded_path = quote(p.path, safe="/:@-._~!$&'()*+,;=%")
        new_query = urlencode(parse_qsl(p.query, keep_blank_values=True), doseq=True, quote_via=quote)
        new_fragment = quote(p.fragment, safe="/:@-._~!$&'()*+,;=%")
        candidate = urlunsplit((p.scheme, p.netloc, encoded_path, new_query, new_fragment))
    else:
        # Relative path: concatenate safely with base
        candidate = concatenate_url_path(base_url, next_param)
        # validate after concatenation
        validate_url(candidate)

    cand_parts = urlparse.urlsplit(candidate)
    cand_host_norm = _normalize_host(cand_parts.hostname or "")

    # Determine if candidate is within base domain
    if cand_host_norm == base_host_norm:
        return candidate

    # If base host is a domain (not IP/localhost), allow subdomains
    base_is_ip = _host_is_ip(base_host_norm)
    if not base_is_ip and base_host_norm != "localhost":
        if _host_matches_domain(cand_host_norm, base_host_norm):
            return candidate

    # Check allowed external domains list (allow their subdomains as well)
    for dom in ALLOWED_DOMAINS:
        dom_norm = _normalize_host(dom)
        if dom_norm and (_host_matches_domain(cand_host_norm, dom_norm)):
            return candidate

    raise ValueError("Redirect target is not allowed")
