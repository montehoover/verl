from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def handle_webhook_request(external_url: str) -> str:
    """
    Processes and validates a webhook URL from an external service.

    Args:
        external_url: The URL string from the external service.

    Returns:
        The validated URL if it meets all criteria.

    Raises:
        ValueError: If the URL's domain is not approved, or if it points
                    to an internal/private network, or if the URL is malformed.
    """
    if not external_url or not isinstance(external_url, str):
        raise ValueError("External URL must be a non-empty string.")

    try:
        parsed_url = urlparse(external_url)
    except Exception as e: # Catch any parsing errors, though urlparse is quite robust
        raise ValueError(f"Malformed URL: {external_url}. Error: {e}")

    # Scheme and netloc (domain part) must be present for a valid absolute URL
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid or relative URL provided: {external_url}. Must be an absolute URL.")

    domain = parsed_url.hostname # Use hostname to ignore port if present

    if domain is None: # Should be caught by netloc check, but good for safety
        raise ValueError(f"Could not extract domain from URL: {external_url}")

    # Check if the domain is in the approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    # Check if the domain points to an internal/private network
    # This check is basic and might need enhancement for more complex scenarios (e.g., CIDR notation)
    for internal_prefix in INTERNAL_IP_RANGES:
        if domain.startswith(internal_prefix):
            raise ValueError(f"URL points to an internal/private network: {external_url}")

    # If all checks pass, return the original URL
    return external_url

if __name__ == '__main__':
    test_urls = [
        "https://api.service.com/data/items",
        "http://hooks.webapp.com/event/new_user",
        "https://api.service.com:8080/path", # Domain check should use hostname
        "ftp://api.service.com/files", # Scheme check (implicitly handled by domain check)
        "https://www.google.com/search",
        "http://localhost/api/status",
        "https://127.0.0.1/admin",
        "http://10.0.0.5/internal/service",
        "https://192.168.1.100/config",
        "malformed-url",
        "/relative/path",
        "http://api.service.com", # Valid
        "http://internal.api.service.com/data", # Not in APPROVED_API_DOMAINS
    ]

    for url in test_urls:
        try:
            validated_url = handle_webhook_request(url)
            print(f"SUCCESS: '{url}' -> Validated: '{validated_url}'")
        except ValueError as e:
            print(f"FAILURE: '{url}' -> Error: {e}")

    print("\nTesting with non-string input:")
    try:
        handle_webhook_request(None)
    except ValueError as e:
        print(f"FAILURE: None -> Error: {e}")

    try:
        handle_webhook_request(12345)
    except ValueError as e:
        print(f"FAILURE: 12345 -> Error: {e}")
