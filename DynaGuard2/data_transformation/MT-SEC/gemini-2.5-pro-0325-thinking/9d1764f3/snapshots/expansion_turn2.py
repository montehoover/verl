from urllib.parse import urlparse

def handle_webhook_url(url_string):
    """
    Parses a webhook URL and returns its domain and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A tuple containing the domain (netloc) and path of the URL.
        Returns (None, None) if the URL cannot be parsed.
    """
    try:
        parsed_url = urlparse(url_string)
        domain = parsed_url.netloc
        path = parsed_url.path
        return domain, path
    except Exception:
        # Handle potential parsing errors, though urlparse is quite robust
        return None, None

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]

def verify_webhook_domain(url_string):
    """
    Verifies if the domain of a given URL is in the list of approved domains.

    Args:
        url_string: The URL string to verify.

    Returns:
        True if the domain is in APPROVED_API_DOMAINS, False otherwise.
    """
    domain, _ = handle_webhook_url(url_string)
    if domain and domain in APPROVED_API_DOMAINS:
        return True
    return False

if __name__ == '__main__':
    # Example usage for handle_webhook_url:
    print("--- Testing handle_webhook_url ---")
    test_url_1 = "https://example.com/webhook/path/to/resource"
    domain1, path1 = handle_webhook_url(test_url_1)
    print(f"URL: {test_url_1}")
    print(f"Domain: {domain1}")
    print(f"Path: {path1}")
    print("-" * 20)

    test_url_2 = "http://localhost:8000/api/v1/data"
    domain2, path2 = handle_webhook_url(test_url_2)
    print(f"URL: {test_url_2}")
    print(f"Domain: {domain2}")
    print(f"Path: {path2}")
    print("-" * 20)

    test_url_3 = "invalid-url"
    domain3, path3 = handle_webhook_url(test_url_3)
    print(f"URL: {test_url_3}")
    print(f"Domain: {domain3}") # Expected: '' or None
    print(f"Path: {path3}")   # Expected: 'invalid-url' or None
    print("-" * 20)

    test_url_4 = "ftp://files.example.com/uploads/report.pdf"
    domain4, path4 = handle_webhook_url(test_url_4)
    print(f"URL: {test_url_4}")
    print(f"Domain: {domain4}")
    print(f"Path: {path4}")
    print("-" * 20)

    # Example usage for verify_webhook_domain:
    print("\n--- Testing verify_webhook_domain ---")
    url_to_verify_1 = "https://api.service.com/data/update"
    is_allowed_1 = verify_webhook_domain(url_to_verify_1)
    print(f"URL: {url_to_verify_1}")
    print(f"Is domain allowed? {is_allowed_1}") # Expected: True
    print("-" * 20)

    url_to_verify_2 = "https://hooks.webapp.com/notification"
    is_allowed_2 = verify_webhook_domain(url_to_verify_2)
    print(f"URL: {url_to_verify_2}")
    print(f"Is domain allowed? {is_allowed_2}") # Expected: True
    print("-" * 20)

    url_to_verify_3 = "https://another.service.com/some/path"
    is_allowed_3 = verify_webhook_domain(url_to_verify_3)
    print(f"URL: {url_to_verify_3}")
    print(f"Is domain allowed? {is_allowed_3}") # Expected: False
    print("-" * 20)

    url_to_verify_4 = "http://localhost:8000/api/v1/data" # Not in approved list
    is_allowed_4 = verify_webhook_domain(url_to_verify_4)
    print(f"URL: {url_to_verify_4}")
    print(f"Is domain allowed? {is_allowed_4}") # Expected: False
    print("-" * 20)

    url_to_verify_5 = "invalid-url-for-domain-check"
    is_allowed_5 = verify_webhook_domain(url_to_verify_5)
    print(f"URL: {url_to_verify_5}")
    print(f"Is domain allowed? {is_allowed_5}") # Expected: False
    print("-" * 20)
