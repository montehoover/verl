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

if __name__ == '__main__':
    # Example Usage
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
