from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def process_hook(hook_url: str) -> str:
    """
    Handles a webhook URL sent by an external service. The resulting URL will be
    used to invoke an internal API action.

    Args:
        hook_url: A string representing the URL provided by the external service.

    Returns:
        A safely validated URL that references an approved API endpoint.

    Raises:
        ValueError: If the URL's domain is outside the list of approved API
                    endpoints or aims at an internal network location.
    """
    parsed_url = urlparse(hook_url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: Could not parse hostname.")

    # Check if the hostname is an IP address and if it's in the internal ranges
    is_internal_ip = False
    for internal_range in INTERNAL_IP_RANGES:
        if hostname == internal_range or (internal_range.endswith('.') and hostname.startswith(internal_range)):
            is_internal_ip = True
            break
    
    if is_internal_ip:
        raise ValueError(f"URL targets an internal network location: {hostname}")

    # Check if the domain is in the approved list
    domain_approved = False
    for approved_domain in APPROVED_API_DOMAINS:
        if hostname == approved_domain or hostname.endswith(f".{approved_domain}"):
            domain_approved = True
            break
    
    if not domain_approved:
        raise ValueError(f"URL domain is not approved: {hostname}")

    return hook_url
