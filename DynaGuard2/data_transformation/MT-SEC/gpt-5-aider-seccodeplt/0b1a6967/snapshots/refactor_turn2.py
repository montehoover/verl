import re
import ipaddress
from dataclasses import dataclass
from typing import Callable, Iterable
from urllib.parse import urlparse, ParseResult

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


# -------- Pure helper functions (domain/IP logic extracted) --------

def to_idna_host(host: str) -> str:
    """
    Normalize hostname: strip trailing dot, lowercase, convert to IDNA ASCII.
    Raises ValueError if encoding fails.
    """
    if not host:
        raise ValueError("URL must include a valid hostname.")
    normalized = host.rstrip(".").lower()
    try:
        return normalized.encode("idna").decode("ascii")
    except Exception as exc:
        raise ValueError("Invalid hostname encoding.") from exc


def matches_internal_pattern(host: str, patterns: Iterable[str]) -> bool:
    """
    Return True if the host matches any of the provided internal IP/host patterns.
    """
    for pattern in patterns:
        if re.match(pattern, host):
            return True
    return False


def is_private_or_special_ip(host: str) -> bool:
    """
    Return True if host is an IP address that is private, loopback, link-local,
    reserved, multicast, or unspecified. Returns False if host is not an IP.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_allowed_domain(host: str, allowed_domains: Iterable[str]) -> bool:
    """
    Return True if host exactly matches one of the allowed domains.
    """
    allowed = {d.lower() for d in allowed_domains}
    return host.lower() in allowed


# -------- Pipeline context and steps --------

@dataclass
class URLContext:
    original_url: str
    parsed: ParseResult
    host: str | None = None


def _step_require_https(ctx: URLContext) -> URLContext:
    if ctx.parsed.scheme.lower() != "https":
        raise ValueError("Only HTTPS URLs are allowed.")
    return ctx


def _step_require_netloc(ctx: URLContext) -> URLContext:
    if not ctx.parsed.netloc:
        raise ValueError("URL must include a network location (host).")
    return ctx


def _step_require_no_credentials(ctx: URLContext) -> URLContext:
    if ctx.parsed.username or ctx.parsed.password:
        raise ValueError("User info in URL is not allowed.")
    return ctx


def _step_normalize_host(ctx: URLContext) -> URLContext:
    host = ctx.parsed.hostname
    if not host:
        raise ValueError("URL must include a valid hostname.")
    ctx.host = to_idna_host(host)
    return ctx


def _step_block_internal_host_patterns(ctx: URLContext) -> URLContext:
    # Host is normalized at this point
    if matches_internal_pattern(ctx.host or "", INTERNAL_IP_PATTERNS):
        raise ValueError("Internal addresses are not allowed.")
    return ctx


def _step_block_internal_ip(ctx: URLContext) -> URLContext:
    if is_private_or_special_ip(ctx.host or ""):
        raise ValueError("Internal or non-routable IP addresses are not allowed.")
    return ctx


def _step_enforce_allowed_domain(ctx: URLContext) -> URLContext:
    if not is_allowed_domain(ctx.host or "", ALLOWED_API_DOMAINS):
        raise ValueError("Unapproved API domain.")
    return ctx


def _run_pipeline(ctx: URLContext) -> URLContext:
    steps: list[Callable[[URLContext], URLContext]] = [
        _step_require_https,
        _step_require_netloc,
        _step_require_no_credentials,
        _step_normalize_host,
        _step_block_internal_host_patterns,
        _step_block_internal_ip,
        _step_enforce_allowed_domain,
    ]
    for step in steps:
        ctx = step(ctx)
    return ctx


def verify_api_address(endpoint_address: str) -> str:
    """
    Verify that the provided endpoint_address is a secure (HTTPS), external API URL
    pointing only to an approved domain and not to any internal networks.

    Args:
        endpoint_address: The URL string to verify.

    Returns:
        The original URL string (trimmed) if it passes verification.

    Raises:
        ValueError: If the URL is invalid, not HTTPS, targets an internal address,
                    or is not in the approved domain list.
    """
    if not isinstance(endpoint_address, str):
        raise ValueError("Endpoint address must be a string.")

    url = endpoint_address.strip()
    parsed = urlparse(url)

    ctx = URLContext(original_url=url, parsed=parsed)
    _run_pipeline(ctx)

    return url
