from urllib.parse import urlparse

def parse_webhook_url(webhook_url: str) -> tuple[str | None, str | None, str | None]:
    """
    Parses a webhook URL and extracts its scheme, hostname, and path.

    Args:
        webhook_url: The URL string to parse.

    Returns:
        A tuple containing the scheme, hostname, and path.
        Returns (None, None, None) if the URL is malformed or cannot be parsed.
    """
    try:
        parsed_url = urlparse(webhook_url)
        scheme = parsed_url.scheme if parsed_url.scheme else None
        hostname = parsed_url.hostname if parsed_url.hostname else None
        path = parsed_url.path if parsed_url.path else None
        return scheme, hostname, path
    except Exception:
        # Broad exception to catch any parsing errors, specific exceptions can be handled if needed
        return None, None, None

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def detect_internal_url(webhook_url: str) -> bool:
    """
    Detects if a URL points to an internal network address.

    Args:
        webhook_url: The URL string to check.

    Returns:
        True if the URL's hostname matches a known internal IP pattern, False otherwise.
    """
    _, hostname, _ = parse_webhook_url(webhook_url)
    if hostname:
        for internal_pattern in INTERNAL_IP_RANGES:
            if hostname == internal_pattern or hostname.startswith(internal_pattern):
                return True
    return False

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]

def process_webhook(webhook_url: str) -> str:
    """
    Processes a webhook URL, ensuring it's not an internal address and belongs to an approved domain.

    Args:
        webhook_url: The URL string to process.

    Returns:
        The validated webhook_url.

    Raises:
        ValueError: If the URL is internal, malformed, or not from an approved domain.
    """
    if detect_internal_url(webhook_url):
        raise ValueError(f"Internal URL detected and not allowed: {webhook_url}")

    _, hostname, _ = parse_webhook_url(webhook_url)

    if not hostname:
        raise ValueError(f"Could not parse hostname from URL: {webhook_url}")

    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"URL domain '{hostname}' is not approved. Approved domains: {APPROVED_API_DOMAINS}")

    return webhook_url

if __name__ == '__main__':
    # Example Usage for parse_webhook_url
    test_url_valid = "https://example.com/webhook/path?query=param"
    scheme, hostname, path = parse_webhook_url(test_url_valid)
    print(f"URL: {test_url_valid}")
    print(f"Scheme: {scheme}, Hostname: {hostname}, Path: {path}")

    test_url_invalid = "this is not a url"
    scheme, hostname, path = parse_webhook_url(test_url_invalid)
    print(f"\nURL: {test_url_invalid}")
    print(f"Scheme: {scheme}, Hostname: {hostname}, Path: {path}")

    test_url_no_path = "ftp://fileserver.example.com"
    scheme, hostname, path = parse_webhook_url(test_url_no_path)
    print(f"\nURL: {test_url_no_path}")
    print(f"Scheme: {scheme}, Hostname: {hostname}, Path: {path}")

    test_url_with_port = "http://localhost:8080/api/v1/hook"
    scheme, hostname, path = parse_webhook_url(test_url_with_port)
    print(f"\nURL: {test_url_with_port}")
    print(f"Scheme: {scheme}, Hostname: {hostname}, Path: {path}")

    # Example Usage for detect_internal_url
    print("\n--- Internal URL Detection ---")
    internal_urls_to_test = [
        "http://localhost/some/path",
        "https://127.0.0.1/api",
        "http://10.0.0.5/webhook",
        "https://192.168.1.100/test",
        "ftp://my-internal-server/files", # Assuming my-internal-server resolves to an internal IP
        "https://example.com/safe",
        "http://172.16.0.1/not-strictly-internal-by-list" # Example of an IP not in our list
    ]

    for url in internal_urls_to_test:
        is_internal = detect_internal_url(url)
        print(f"URL: {url} -> Is Internal? {is_internal}")

    print(f"URL: {test_url_invalid} -> Is Internal? {detect_internal_url(test_url_invalid)}")

    # Example Usage for process_webhook
    print("\n--- Webhook Processing ---")
    urls_to_process = [
        "https://api.service.com/v1/update",
        "http://hooks.webapp.com/payload",
        "https://user.api.service.com/data", # Subdomain, should fail if not explicitly in APPROVED_API_DOMAINS
        "https://another.service.com/event",
        "http://localhost/trigger",
        "https://127.0.0.1/action",
        "this is not a url for processing"
    ]

    for url in urls_to_process:
        print(f"\nProcessing URL: {url}")
        try:
            processed_url = process_webhook(url)
            print(f"Successfully processed: {processed_url}")
        except ValueError as e:
            print(f"Error: {e}")
