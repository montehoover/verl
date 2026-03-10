from urllib.parse import urlparse

APPROVED_DOMAINS = ['example.com', 'myapp.com', 'trusted.com']

def is_valid_domain(url: str) -> bool:
    """
    Validates if the domain of a given URL is in the list of approved domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's domain is in APPROVED_DOMAINS, False otherwise.
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Handle cases like 'www.example.com' if 'example.com' is approved
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain in APPROVED_DOMAINS
    except Exception:
        # In case of any parsing error, consider it invalid
        return False

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/page",
        "https://www.myapp.com/another/page?query=param",
        "ftp://trusted.com",
        "http://sub.example.com/path", # This will be False unless sub.example.com is added
        "http://untrusted.com",
        "example.com", # This will be False as it's not a full URL and urlparse might not get netloc
        "http://www.example.com",
        "https://example.com",
        "http://another-example.com",
        "not_a_url"
    ]

    for test_url in test_urls:
        print(f"URL: {test_url}, Is Valid Domain: {is_valid_domain(test_url)}")

    print("\nTesting with a domain that should be approved if www is stripped:")
    print(f"URL: http://www.trusted.com, Is Valid Domain: {is_valid_domain('http://www.trusted.com')}")

    print("\nTesting with a subdomain that is not explicitly approved:")
    print(f"URL: http://user.myapp.com, Is Valid Domain: {is_valid_domain('http://user.myapp.com')}")
