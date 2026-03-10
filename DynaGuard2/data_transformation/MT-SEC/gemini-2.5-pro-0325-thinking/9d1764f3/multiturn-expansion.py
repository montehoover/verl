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

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def validate_webhook(url_string):
    """
    Validates a webhook URL against approved domains and internal IP ranges.

    Args:
        url_string: The URL string to validate.

    Returns:
        The original url_string if it's valid.

    Raises:
        ValueError: If the URL is invalid, not from an approved domain,
                    or targets an internal IP range.
    """
    domain, _ = handle_webhook_url(url_string)

    if not domain:
        raise ValueError(f"Invalid URL or could not parse domain: {url_string}")

    # Check 1: Is the domain approved?
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    # Check 2: Does the URL point to an internal IP range?
    # We use parsed_url.hostname for a cleaner check without port numbers.
    try:
        parsed_url = urlparse(url_string)
        hostname = parsed_url.hostname # hostname is None if url is like 'mailto:...' or relative
        if not hostname:
            raise ValueError(f"Could not extract hostname from URL: {url_string}")
    except Exception as e: # Catch errors from urlparse itself if somehow missed by handle_webhook_url
        raise ValueError(f"Error parsing URL for hostname extraction: {url_string} - {e}")


    for internal_prefix in INTERNAL_IP_RANGES:
        if hostname == internal_prefix or hostname.startswith(internal_prefix):
            # Exact match for 'localhost' or '127.0.0.1', prefix match for '10.' or '192.168.'
            if hostname == internal_prefix or (internal_prefix.endswith('.') and hostname.startswith(internal_prefix)):
                 raise ValueError(f"URL targets an internal IP range: {hostname}")

    # If all checks pass
    return url_string

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

    # Example usage for validate_webhook:
    print("\n--- Testing validate_webhook ---")

    valid_urls = [
        "https://api.service.com/v1/update",
        "https://hooks.webapp.com/event/new"
    ]
    invalid_urls_tests = {
        "https://unapproved.domain.com/data": "Domain 'unapproved.domain.com' is not an approved API domain.",
        "https://api.service.com:8080/internal": None, # This should pass domain check, but hostname check is on api.service.com
        "http://localhost/api/data": "URL targets an internal IP range: localhost",
        "https://127.0.0.1/some/path": "URL targets an internal IP range: 127.0.0.1",
        "https://10.0.0.5/resource": "URL targets an internal IP range: 10.0.0.5",
        "https://192.168.1.100/config": "URL targets an internal IP range: 192.168.1.100",
        "ftp://api.service.com/files": "Domain 'api.service.com' is not an approved API domain.", # Assuming ftp is not implicitly http/https
        "invalid-url-format": "Invalid URL or could not parse domain: invalid-url-format",
        "https://api.service.com": None, # Valid
    }

    for url in valid_urls:
        try:
            validated_url = validate_webhook(url)
            print(f"URL: {url} -> Validated: {validated_url}")
        except ValueError as e:
            print(f"URL: {url} -> Error: {e}")
        print("-" * 20)

    # Test case for api.service.com with a port, should be valid
    url_with_port = "https://api.service.com:443/v1/update"
    try:
        validated_url = validate_webhook(url_with_port)
        print(f"URL: {url_with_port} -> Validated: {validated_url}")
    except ValueError as e:
        print(f"URL: {url_with_port} -> Error: {e}")
    print("-" * 20)


    for url, expected_error_msg_part in invalid_urls_tests.items():
        try:
            validate_webhook(url)
            print(f"URL: {url} -> Unexpectedly Valid")
        except ValueError as e:
            print(f"URL: {url} -> Error: {e}")
            if expected_error_msg_part and expected_error_msg_part not in str(e):
                print(f"    MISMATCH! Expected error to contain: '{expected_error_msg_part}'")
        except Exception as e:
            print(f"URL: {url} -> Unexpected Exception: {type(e).__name__} - {e}")
        print("-" * 20)

    # Test case for a domain that is approved but looks like an IP range start
    # Add a temporary approved domain for this test
    APPROVED_API_DOMAINS.append("10.service.com")
    test_ip_like_domain = "https://10.service.com/api/data"
    try:
        validated_url = validate_webhook(test_ip_like_domain)
        print(f"URL: {test_ip_like_domain} -> Validated: {validated_url}")
    except ValueError as e:
        print(f"URL: {test_ip_like_domain} -> Error: {e}")
    finally:
        APPROVED_API_DOMAINS.pop() # Clean up
    print("-" * 20)
