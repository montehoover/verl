import os
import ipaddress
from urllib.parse import urlsplit
from typing import Iterable, Set

__all__ = ["validate_url", "set_trusted_domains"]

# Internal storage for trusted entities
_TRUSTED_DOMAINS_IDNA: Set[str] = set()  # e.g., "example.com" in IDNA ASCII (punycode when applicable)
_TRUSTED_IPS: Set[str] = set()           # canonical string representation of IPs (IPv4 or IPv6)


def _to_idna(domain: str) -> str:
    """
    Normalize a Unicode domain to its IDNA ASCII (punycode) form, lowercased and without trailing dot.
    """
    domain = domain.strip().strip(".").lower()
    # If it parses as an IP, just return canonical IP text
    try:
        return ipaddress.ip_address(domain).compressed
    except ValueError:
        pass
    # Domain to IDNA ASCII
    return domain.encode("idna").decode("ascii")


def set_trusted_domains(domains: Iterable[str]) -> None:
    """
    Configure the trusted domains/IPs list used by validate_url.
    Each item can be:
      - a domain (e.g., "example.com", "sub.example.com")
      - an IP address (IPv4 or IPv6, e.g., "192.0.2.1", "2001:db8::1")
    For domains, subdomains of a trusted domain are also considered trusted.
    """
    global _TRUSTED_DOMAINS_IDNA, _TRUSTED_IPS
    doms: Set[str] = set()
    ips: Set[str] = set()
    for d in domains:
        if not isinstance(d, str):
            continue
        dd = d.strip()
        if not dd:
            continue
        # If IP, store canonical form in IP set; else store IDNA domain in domain set
        try:
            ip = ipaddress.ip_address(dd)
            ips.add(ip.compressed)
            continue
        except ValueError:
            pass
        doms.add(_to_idna(dd))
    _TRUSTED_DOMAINS_IDNA = doms
    _TRUSTED_IPS = ips


def _init_trusted_from_env() -> None:
    """
    Initialize trusted domains/IPs from TRUSTED_DOMAINS env var, comma-separated.
    Example: TRUSTED_DOMAINS="example.com,sub.example.com,192.0.2.10,2001:db8::1"
    """
    env_val = os.getenv("TRUSTED_DOMAINS", "")
    if env_val:
        parts = [p.strip() for p in env_val.split(",")]
        set_trusted_domains(p for p in parts if p)


_init_trusted_from_env()


def validate_url(url: str) -> bool:
    """
    Validate a URL and return True if it is valid and belongs to a trusted domain/IP.
    Returns:
      - True if the URL is syntactically valid AND host is within trusted domains/IPs
      - False if the URL is syntactically valid BUT host is not trusted
    Raises:
      - ValueError if the URL is syntactically invalid or unsafe
    Security considerations:
      - Only http and https schemes are allowed
      - Userinfo (username:password@) is rejected
      - Newlines and backslashes are rejected to prevent injection/ambiguities
      - Hostnames are normalized to IDNA for matching; IPs are canonicalized
    """
    if not isinstance(url, str) or not url:
        raise ValueError("URL must be a non-empty string")

    # Basic sanitation to prevent header/filepath injection issues
    if any(ch in url for ch in ("\r", "\n", "\x00")):
        raise ValueError("URL contains forbidden control characters")
    if "\\" in url:
        # Disallow backslashes to avoid ambiguous parsing or \\@ tricks
        raise ValueError("URL contains forbidden backslash characters")

    parsed = urlsplit(url)

    # Scheme must be http or https
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError("Unsupported or missing URL scheme")

    # Must be absolute with a network location
    if not parsed.netloc:
        raise ValueError("URL must be absolute with a network location")

    # Disallow userinfo (username:password@host)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Userinfo in URL is not allowed")

    # Validate/normalize port (will raise ValueError if invalid)
    try:
        _ = parsed.port  # Accessing .port validates it's numeric and within 0-65535
    except ValueError as exc:
        raise ValueError("Invalid port in URL") from exc

    # Extract and normalize hostname
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no hostname")

    # Normalize to canonical representation
    norm_host = _to_idna(host)

    # Determine if host is an IP
    is_ip = False
    try:
        ip_obj = ipaddress.ip_address(norm_host)
        is_ip = True
        norm_host = ip_obj.compressed
    except ValueError:
        is_ip = False

    # Evaluate trust
    if is_ip:
        return norm_host in _TRUSTED_IPS

    # Domain trust: exact match or subdomain match
    if not _TRUSTED_DOMAINS_IDNA:
        return False

    for trusted in _TRUSTED_DOMAINS_IDNA:
        if norm_host == trusted or norm_host.endswith("." + trusted):
            return True

    return False
