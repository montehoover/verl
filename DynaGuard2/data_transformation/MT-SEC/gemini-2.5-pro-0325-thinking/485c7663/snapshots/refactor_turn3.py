"""
This module provides functionality to validate API endpoint URLs, ensuring they
are secure and do not point to internal or disallowed resources. This helps
mitigate risks such as Server-Side Request Forgery (SSRF).

The main function, `enforce_api_security`, checks:
1.  URL validity and scheme (must be HTTPS).
2.  If the hostname is an internal IP address.
3.  If the hostname is an allowed domain or a forbidden IP address.
"""
import re
from urllib.parse import urlparse

# List of fully qualified domain names that are approved for API calls.
ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]

# Regular expression patterns to identify internal or private IP addresses.
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
    """
    Performs a basic check to see if a hostname string resembles an IP address.
    This is a preliminary check and does not validate the IP address's correctness.

    Args:
        hostname: The hostname string to check.

    Returns:
        True if the hostname consists only of digits and dots, False otherwise.
    """
    return all(c.isdigit() or c == '.' for c in hostname)

def _validate_hostname_ip_format(hostname: str):
    """
    Validates the format of a hostname if it appears to be an IP address,
    using the `ipaddress` module if available.

    This function should ideally be called after `_is_hostname_potentially_ip`
    returns True for the given hostname.

    Args:
        hostname: The hostname string (potentially an IP address) to validate.

    Raises:
        ValueError: If the `ipaddress` module is available and the hostname
                    is not a valid IP address format.
    """
    if _IPADDRESS_MODULE_AVAILABLE:
        try:
            ipaddress.ip_address(hostname)
        except ValueError: # Catches ipaddress.AddressValueError from ipaddress module
            raise ValueError(f"Invalid IP address format in hostname: '{hostname}'. "
                             "Rejected due to malformed IP.")

def _check_internal_ip(hostname: str):
    """
    Checks if the given hostname matches any predefined internal IP patterns.

    Args:
        hostname: The hostname to check.

    Raises:
        ValueError: If the hostname matches an internal IP pattern, indicating
                    it's an attempt to access a restricted internal resource.
    """
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address '{hostname}' is forbidden. "
                             "Rejected due to internal IP policy.")

def _check_domain_and_ip_policy(hostname: str):
    """
    Validates the hostname against the list of allowed domains and ensures
    it's not a direct IP access unless that IP's domain is allowed (which
    is implicitly handled by checking `ALLOWED_API_DOMAINS` first).

    This function assumes that checks for internal IPs (`_check_internal_ip`)
    have already been performed.

    Args:
        hostname: The hostname to validate.

    Raises:
        ValueError: If the hostname is not in `ALLOWED_API_DOMAINS` or if it's
                    an IP address not corresponding to an allowed domain.
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
        raise ValueError(f"Direct IP access to '{hostname}' is forbidden. "
                         "Only allowed domains can be accessed.")
    else:
        # It's a domain name (not an IP format) and not in ALLOWED_API_DOMAINS.
        raise ValueError(f"Domain '{hostname}' is not allowed. "
                         f"Allowed domains are: {', '.join(ALLOWED_API_DOMAINS)}. "
                         "Rejected due to domain policy.")


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
    except Exception as e: # Broad exception to catch any parsing errors
        raise ValueError(f"Invalid URL format: '{api_endpoint}'. Parsing failed. Error: {e}")

    if not parsed_url.scheme:
        raise ValueError(f"URL scheme is missing in '{api_endpoint}'. "
                         "A scheme (e.g., 'https') is required.")

    if parsed_url.scheme.lower() != "https":
        raise ValueError(f"URL scheme must be HTTPS. Received '{parsed_url.scheme}' for URL: '{api_endpoint}'. "
                         "Rejected due to insecure scheme.")

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"Hostname could not be determined from URL: '{api_endpoint}'. "
                         "A valid hostname is required.")

    # Perform security checks on the hostname
    _check_internal_ip(hostname) # Raises ValueError if internal IP
    _check_domain_and_ip_policy(hostname)

    return api_endpoint
