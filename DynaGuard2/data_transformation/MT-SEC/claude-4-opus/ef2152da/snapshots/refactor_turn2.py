from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def _extract_hostname(url):
    """Extract hostname from URL."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname found")
    return parsed, hostname

def _is_internal_network(hostname):
    """Check if hostname refers to internal/private network."""
    for internal_prefix in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_prefix):
            return True
    return False

def _is_approved_domain(hostname):
    """Check if hostname is in the approved domains list."""
    return hostname in APPROVED_API_DOMAINS

def _is_secure_protocol(parsed_url):
    """Check if URL uses HTTPS protocol."""
    return parsed_url.scheme == 'https'

def _validate_hostname_security(hostname):
    """Validate that hostname is not accessing internal networks."""
    if _is_internal_network(hostname):
        raise ValueError(f"URL attempts to access internal/private network: {hostname}")

def _validate_domain_approval(hostname):
    """Validate that hostname is in approved domains."""
    if not _is_approved_domain(hostname):
        raise ValueError(f"URL domain not in approved list: {hostname}")

def _validate_protocol_security(parsed_url):
    """Validate that URL uses secure protocol."""
    if not _is_secure_protocol(parsed_url):
        raise ValueError("URL must use HTTPS protocol")

def validate_callback(callback_url):
    """
    Validates a webhook URL from an external source.
    
    Args:
        callback_url: a string representing the external webhook URL
        
    Returns:
        A validated and secure URL that is allowed for internal API interactions
        
    Raises:
        ValueError: if the URL domain is not from the approved API endpoint list
                   or if it tries to access internal/private network addresses
    """
    # Extract URL components
    parsed, hostname = _extract_hostname(callback_url)
    
    # Perform validations
    _validate_hostname_security(hostname)
    _validate_domain_approval(hostname)
    _validate_protocol_security(parsed)
    
    # Return the validated URL
    return callback_url
