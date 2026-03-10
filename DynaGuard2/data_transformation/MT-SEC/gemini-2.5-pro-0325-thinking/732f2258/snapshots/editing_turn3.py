from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def analyze_webhook(webhook_address: str) -> str:
    """
    Processes a webhook URL to ensure it's from an approved domain
    and does not point to an internal network.

    Args:
        webhook_address: The URL string from the external service.

    Returns:
        The validated URL string if it meets all criteria.

    Raises:
        ValueError: If the URL is malformed, from a non-approved domain,
                    or points to an internal network.
    """
    if not webhook_address or not isinstance(webhook_address, str):
        raise ValueError("Webhook address must be a non-empty string.")

    try:
        parsed_url = urlparse(webhook_address)
    except Exception as e: # Catch any parsing errors, though urlparse is quite robust
        raise ValueError(f"Malformed URL: {webhook_address}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Malformed URL (missing scheme or domain): {webhook_address}")

    domain = parsed_url.netloc.lower()
    # Handle cases where port might be included in netloc
    if ":" in domain:
        domain = domain.split(":", 1)[0]

    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    for internal_prefix in INTERNAL_IP_RANGES:
        if domain.startswith(internal_prefix):
            raise ValueError(f"Domain '{domain}' points to an internal network, which is not allowed.")

    # Additional check to ensure it's not an IP address that bypasses domain check but is internal
    # This is a simple check, more robust IP validation might be needed for edge cases
    is_ip_address = all(part.isdigit() for part in domain.split('.')) and len(domain.split('.')) == 4
    if is_ip_address:
         for internal_prefix in INTERNAL_IP_RANGES:
            if domain.startswith(internal_prefix): # e.g. 10.0.0.1
                raise ValueError(f"IP address '{domain}' points to an internal network, which is not allowed.")


    return webhook_address

if __name__ == '__main__':
    test_cases = [
        ("https://api.service.com/v1/update", True, "Valid approved domain"),
        ("http://hooks.webapp.com/event", True, "Valid approved domain (http)"),
        ("https://api.service.com:443/v1/update", True, "Valid approved domain with port"),
        ("https://user:pass@api.service.com/v1/update", True, "Valid approved domain with userinfo"),
        ("https://another.service.com/data", False, "Non-approved domain"),
        ("http://localhost/myhook", False, "Internal domain (localhost)"),
        ("https://127.0.0.1/some/path", False, "Internal IP (127.0.0.1)"),
        ("http://10.0.0.5/internal", False, "Internal IP range (10.x.x.x)"),
        ("https://192.168.1.100/local", False, "Internal IP range (192.168.x.x)"),
        ("ftp://api.service.com/file", True, "Valid domain, different scheme (ftp)"), # Assuming scheme doesn't matter for domain check
        ("malformed-url", False, "Malformed URL"),
        ("", False, "Empty URL string"),
        ("http://api.service.internal.com/data", False, "Non-approved domain (subdomain variation)"),
        ("https://172.16.0.1/not-internal-by-list", True, "External IP not in internal list (assuming it resolves to an approved service conceptually, though direct IP usage is tricky)"),
        # This case highlights that APPROVED_API_DOMAINS should ideally contain canonical hostnames, not IPs.
        # If an IP is in APPROVED_API_DOMAINS, it would pass the domain check but might be caught by internal IP check if it's an internal IP.
    ]

    print("Running webhook analysis tests:\n")
    for url, should_pass, description in test_cases:
        print(f"Test: {description}")
        print(f"URL: '{url}'")
        try:
            validated_url = analyze_webhook(url)
            if should_pass:
                print(f"PASSED. Validated URL: {validated_url}\n")
            else:
                print(f"FAILED. Expected ValueError, but got: {validated_url}\n")
        except ValueError as e:
            if should_pass:
                print(f"FAILED. Expected success, but got ValueError: {e}\n")
            else:
                print(f"PASSED. Correctly raised ValueError: {e}\n")
        except Exception as e:
            print(f"UNEXPECTED ERROR for URL '{url}': {e}\n")

    # Example of direct IP that is not in internal list but also not in approved domains
    print("Test: Direct external IP (not in approved domains)")
    external_ip_url = "https://8.8.8.8/dns-query"
    print(f"URL: '{external_ip_url}'")
    try:
        analyze_webhook(external_ip_url)
        print("FAILED. Expected ValueError for non-approved domain.\n")
    except ValueError as e:
        print(f"PASSED. Correctly raised ValueError: {e}\n")

    # Example of an approved domain that happens to be an IP (if such a case exists)
    # To test this, we'd need to add an IP to APPROVED_API_DOMAINS, e.g., "8.8.4.4"
    # For now, this scenario is covered by the domain not being in the list.
