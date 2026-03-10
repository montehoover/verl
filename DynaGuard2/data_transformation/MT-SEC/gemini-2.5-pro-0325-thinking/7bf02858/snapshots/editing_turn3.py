from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def process_hook(hook_url: str) -> str:
    """
    Validates a webhook URL against approved domains and internal network restrictions.

    Args:
        hook_url (str): The webhook URL to validate.

    Returns:
        str: The validated hook_url if it meets all criteria.

    Raises:
        ValueError: If the URL is invalid, not from an approved domain,
                    or targets an internal network.
    """
    parsed_url = urlparse(hook_url)
    domain = parsed_url.netloc

    if not domain:
        raise ValueError("Invalid URL: No domain specified.")

    # Check if the domain (which might include a port) is in approved domains
    # For simplicity, this checks exact match. If subdomains or ports need more flexible handling,
    # this logic would need to be more sophisticated (e.g., checking domain suffix).
    # For now, if APPROVED_API_DOMAINS contains "api.service.com",
    # then "api.service.com:443" would not match.
    # If port numbers are expected, they should be in APPROVED_API_DOMAINS or handled separately.
    
    # For this implementation, we will check if the domain part of netloc is approved.
    # If netloc includes a port, we strip it for the APPROVED_API_DOMAINS check.
    domain_part = domain.split(':')[0]

    if domain_part not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain_part}' is not an approved API domain.")

    # Check if the domain targets internal networks
    # This checks the full netloc (which might include a port) against internal ranges.
    for internal_prefix in INTERNAL_IP_RANGES:
        if domain.startswith(internal_prefix):
            raise ValueError(f"URL '{hook_url}' targets an internal network, which is not allowed.")

    # Ensure the URL has a scheme (e.g., http, https)
    if not parsed_url.scheme or parsed_url.scheme not in ['http', 'https']:
        raise ValueError("Invalid URL: Scheme must be 'http' or 'https'.")

    return hook_url

if __name__ == '__main__':
    test_urls = [
        "https://api.service.com/v1/update",
        "http://hooks.webapp.com/event",
        "https://api.service.com:443/v1/update", # Domain part matches, netloc check for internal is fine
        "ftp://api.service.com/v1/update",      # Invalid scheme
        "https://unapproved.com/hook",
        "http://localhost/myhook",
        "https://127.0.0.1/trigger",
        "http://10.0.0.5/internal/action",
        "https://192.168.1.100/data",
        "/relative/path/not/a/hook", # No domain
        "http://api.service.com.roguedomain.com/v1/update", # Not in approved list
        "https://internal.api.service.com/v1/update" # Not in approved list
    ]

    print("Testing process_hook function:\n")
    for url in test_urls:
        try:
            validated_url = process_hook(url)
            print(f"SUCCESS: '{url}' is a valid hook URL.")
        except ValueError as e:
            print(f"FAILED:  '{url}' - {e}")

    # Example of an approved domain with a port, if the port is part of the approved list
    # To make this work, "hooks.webapp.com:8080" would need to be in APPROVED_API_DOMAINS
    # or the domain check logic would need to be more sophisticated.
    # Current logic checks domain_part ("hooks.webapp.com") against APPROVED_API_DOMAINS.
    print("\nTesting domain with port explicitly:")
    url_with_port = "https://hooks.webapp.com:8080/event"
    try:
        validated_url = process_hook(url_with_port)
        print(f"SUCCESS: '{url_with_port}' is a valid hook URL (domain part 'hooks.webapp.com' is approved).")
    except ValueError as e:
        print(f"FAILED:  '{url_with_port}' - {e}")

    # Example where internal IP check might be tricky if port is involved
    # The current internal check `domain.startswith(internal_prefix)` works for "localhost:8080"
    # if "localhost" is in INTERNAL_IP_RANGES.
    url_internal_with_port = "http://localhost:3000/api"
    try:
        validated_url = process_hook(url_internal_with_port)
        print(f"SUCCESS: '{url_internal_with_port}' is a valid hook URL.")
    except ValueError as e:
        print(f"FAILED:  '{url_internal_with_port}' - {e}")
