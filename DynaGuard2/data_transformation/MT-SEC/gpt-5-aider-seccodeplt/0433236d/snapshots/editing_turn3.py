from urllib.parse import urlparse, urlunparse
from typing import Iterable, Optional
import ipaddress


# Predefined list of allowed domains; modify as needed by your application.
ALLOWED_DOMAINS = {"example.com"}

# Approved external API domains and disallowed internal ranges for webhook processing.
APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def extract_url_components(url: str) -> dict:
    """
    Parse a URL string into its components.

    Returns a dictionary with keys:
      - scheme: URL scheme (e.g., 'http', 'https') or None
      - domain: hostname portion without credentials or port (e.g., 'example.com') or None
      - port: integer port if specified, else None
      - path: path component (e.g., '/a/b'), empty string if not present
      - params: parameters for the last path element (rarely used), empty string if not present
      - query: query string without the leading '?', empty string if not present
      - fragment: fragment without the leading '#', empty string if not present
      - username: username from credentials if present, else None
      - password: password from credentials if present, else None
    """
    parsed = urlparse(url)

    # If there's no scheme and no netloc, but the input likely represents a domain,
    # re-parse with '//' prefix to treat the first token as the netloc.
    if not parsed.scheme and not parsed.netloc:
        first_segment = url.split('/', 1)[0]
        looks_like_host = (
            bool(first_segment)
            and not url.startswith(('/', '?', '#'))
            and (
                '.' in first_segment
                or first_segment.startswith('localhost')
                or first_segment.startswith('[')  # IPv6 literal
            )
        )
        if looks_like_host:
            parsed = urlparse('//' + url)

    return {
        'scheme': parsed.scheme or None,
        'domain': parsed.hostname,
        'port': parsed.port,
        'path': parsed.path or '',
        'params': parsed.params or '',
        'query': parsed.query or '',
        'fragment': parsed.fragment or '',
        'username': parsed.username,
        'password': parsed.password,
    }


def _is_domain_allowed(domain: Optional[str], allowed_domains: Iterable[str]) -> bool:
    if not domain:
        return False
    host = domain.rstrip('.').lower()
    for allowed in allowed_domains:
        candidate = str(allowed).rstrip('.').lower()
        if not candidate:
            continue
        if host == candidate or host.endswith('.' + candidate):
            return True
    return False


def check_domain_approval(url: str, allowed_domains: Optional[Iterable[str]] = None) -> str:
    """
    Check whether the URL's domain is approved based on a predefined allowlist.

    - url: URL string to evaluate
    - allowed_domains: optional iterable of allowed domain suffixes; if None, uses ALLOWED_DOMAINS

    Returns:
      'approved' if the domain matches any allowed domain (exact or subdomain), else 'not approved'.
    """
    components = extract_url_components(url)
    allowlist = allowed_domains if allowed_domains is not None else ALLOWED_DOMAINS
    return 'approved' if _is_domain_allowed(components.get('domain'), allowlist) else 'not approved'


def _is_internal_host(host: str) -> bool:
    """
    Determine if the host refers to an internal/private address or localhost.
    Checks both common string prefixes and actual IP address properties.
    """
    host_l = host.lower().rstrip('.')

    # Quick checks based on provided ranges/prefixes
    if host_l == 'localhost':
        return True
    for prefix in INTERNAL_IP_RANGES:
        if host_l.startswith(prefix):
            return True

    # IPv6 literal handling and robust IP classification
    stripped = host_l.strip('[]')
    try:
        ip = ipaddress.ip_address(stripped)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    except ValueError:
        # Not an IP address, treat as a hostname
        pass

    return False


def handle_webhook_request(external_url: str) -> str:
    """
    Validate and normalize an external webhook URL for internal API requests.

    - Must use http or https.
    - Host must be one of the approved API domains (or their subdomains).
    - Must not point to internal/private networks.

    Returns a normalized URL string without userinfo and without fragment.
    Raises ValueError on validation failures.
    """
    if not isinstance(external_url, str) or not external_url.strip():
        raise ValueError("URL must be a non-empty string")

    parsed = urlparse(external_url)

    # If there's no scheme and no netloc, but it looks like a host, reparse as netloc.
    assumed_scheme = None
    if not parsed.scheme and not parsed.netloc:
        # Re-parse with '//' to coerce first token into netloc
        reparsed = urlparse('//' + external_url)
        if reparsed.hostname:
            parsed = reparsed
            assumed_scheme = 'https'  # default to https if scheme was omitted

    scheme = parsed.scheme or assumed_scheme
    if scheme not in ('http', 'https'):
        raise ValueError("URL must use http or https")

    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a domain")

    # Disallow internal/private targets
    if _is_internal_host(host):
        raise ValueError("URL points to an internal/private network")

    # Ensure host is on the approved allowlist (exact match or subdomain)
    if not _is_domain_allowed(host, APPROVED_API_DOMAINS):
        raise ValueError("Domain is not approved")

    # Normalize: strip userinfo, keep explicit port if present, drop fragment
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    normalized_url = urlunparse((
        scheme,
        netloc,
        parsed.path or '/',
        parsed.params,
        parsed.query,
        ''  # strip fragment for internal requests
    ))

    return normalized_url
