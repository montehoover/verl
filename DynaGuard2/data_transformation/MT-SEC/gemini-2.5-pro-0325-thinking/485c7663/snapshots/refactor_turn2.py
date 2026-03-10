import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

try:
    import ipaddress
    _IPADDRESS_MODULE_AVAILABLE = True
except ImportError:
    _IPADDRESS_MODULE_AVAILABLE = False


def _is_hostname_potentially_ip(hostname: str) -> bool:
    """Checks if a hostname string has the basic format of an IP address."""
    return all(c.isdigit() or c == '.' for c in hostname)

def _validate_hostname_ip_format(hostname: str):
    """
    Validates the format of a hostname if it's an IP, using ipaddress module.
    Raises ValueError if the format is invalid.
    This function should only be called if _is_hostname_potentially_ip(hostname) is true.
    Does nothing if ipaddress module is not available.
    """
    if _IPADDRESS_MODULE_AVAILABLE:
        try:
            ipaddress.ip_address(hostname)
        except ValueError: # Catches ipaddress.AddressValueError from ipaddress module
            raise ValueError(f"Invalid IP address format in hostname: {hostname}")

def _check_internal_ip(hostname: str):
    """Raises ValueError if hostname matches internal IP patterns."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address is forbidden: {hostname}")

def _check_domain_and_ip_policy(hostname: str):
    """
    Raises ValueError if hostname is not an allowed domain or is a forbidden IP.
    Assumes internal IP checks have already passed.
    """
    if hostname in ALLOWED_API_DOMAINS:
        return  # Hostname is explicitly allowed

    # If not in allowed domains, check if it's an IP or a disallowed domain
    if _is_hostname_potentially_ip(hostname):
        # It looks like an IP. Validate its format more strictly if possible.
        _validate_hostname_ip_format(hostname) # May raise ValueError for invalid formats
        
        # If _validate_hostname_ip_format didn't raise, it's either a valid IP format
        # or ipaddress module wasn't available (relying on basic check).
        # Since it's an IP (or looks like one) and not in ALLOWED_API_DOMAINS, it's forbidden.
        raise ValueError(f"Direct IP access is forbidden unless domain is allowed: {hostname}")
    else:
        # It's a domain name (not an IP format) and not in ALLOWED_API_DOMAINS.
        raise ValueError(f"Domain not allowed: {hostname}. Allowed domains are: {', '.join(ALLOWED_API_DOMAINS)}")


def enforce_api_security(api_endpoint: str) -> str:
    """
    Ensures a URL points to a secure, external API endpoint.

    Args:
        api_endpoint: The URL to verify.

    Returns:
        The verified URL if allowed.

    Raises:
        ValueError: If the URL is invalid, not HTTPS, targets an unapproved
                    domain, or targets an internal network address.
    """
    try:
        parsed_url = urlparse(api_endpoint)
    except Exception as e:
        raise ValueError(f"Invalid URL: {api_endpoint}. Error: {e}")

    if not parsed_url.scheme:
        raise ValueError(f"URL scheme is missing: {api_endpoint}")

    if parsed_url.scheme.lower() != "https":
        raise ValueError(f"URL must use HTTPS: {api_endpoint}")

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"Hostname could not be determined from URL: {api_endpoint}")

    _check_internal_ip(hostname)
    _check_domain_and_ip_policy(hostname)

    return api_endpoint
