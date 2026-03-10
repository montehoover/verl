from urllib.parse import urlparse

APPROVED_DOMAINS = ['example.com', 'trusted.com']

def is_valid_domain(url: str) -> bool:
    """
    Checks if the domain of the given URL is in the list of approved domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the domain is approved, False otherwise.
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Remove www. if present, as APPROVED_DOMAINS might not have it
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain in APPROVED_DOMAINS
    except Exception:
        # Handle potential parsing errors for malformed URLs
        return False

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/page",
        "https://www.trusted.com/another/page",
        "http://sub.example.com/resource", # This will be False unless sub.example.com is added
        "ftp://untrusted.net/file",
        "http://example.com",
        "https://trusted.com",
        "malformed-url"
    ]

    for test_url in test_urls:
        print(f"URL: {test_url}, Is Valid Domain: {is_valid_domain(test_url)}")

    print("\nTesting with www prefix variations:")
    APPROVED_DOMAINS.append("www.another-trusted.com") # Add with www for testing
    print(f"Is 'http://another-trusted.com' valid? {is_valid_domain('http://another-trusted.com')}") # Should be False
    print(f"Is 'http://www.another-trusted.com' valid? {is_valid_domain('http://www.another-trusted.com')}") # Should be True

    # Test case where approved domain does not have www but URL does
    print(f"Is 'https://www.example.com/path' valid? {is_valid_domain('https://www.example.com/path')}") # Should be True

    # Test case where approved domain has www but URL does not (less common, but good to be aware)
    # Current logic handles this by stripping www from URL, so if 'www.domain.com' is approved,
    # 'domain.com' in URL will be checked against 'www.domain.com' (False)
    # and 'www.domain.com' in URL will be checked against 'www.domain.com' (True after stripping www from URL and comparing to 'www.domain.com')
    # Let's refine APPROVED_DOMAINS to not have 'www.' for consistency or handle it symmetrically.
    # For now, the current logic is: strip www from URL, then check.
    # So, if 'www.explicit-www.com' is in APPROVED_DOMAINS,
    # is_valid_domain('http://explicit-www.com') -> 'explicit-www.com' in APPROVED_DOMAINS -> False
    # is_valid_domain('http://www.explicit-www.com') -> 'explicit-www.com' in APPROVED_DOMAINS -> False
    # This means APPROVED_DOMAINS should store domains without 'www.'
    # Let's adjust the example and logic slightly for clarity.

    # Re-adjusting for clarity: APPROVED_DOMAINS should store base domains.
    # The function will strip 'www.' from the input URL's domain before checking.
    print("\nRefined test for 'www.' handling (APPROVED_DOMAINS should be base domains):")
    # Ensure APPROVED_DOMAINS are base domains for this test
    if 'www.another-trusted.com' in APPROVED_DOMAINS:
        APPROVED_DOMAINS.remove('www.another-trusted.com')
    if 'another-trusted.com' not in APPROVED_DOMAINS:
        APPROVED_DOMAINS.append('another-trusted.com')

    print(f"Current APPROVED_DOMAINS: {APPROVED_DOMAINS}")
    print(f"Is 'http://another-trusted.com' valid? {is_valid_domain('http://another-trusted.com')}")
    print(f"Is 'http://www.another-trusted.com' valid? {is_valid_domain('http://www.another-trusted.com')}")
    print(f"Is 'https://example.com/test' valid? {is_valid_domain('https://example.com/test')}")
    print(f"Is 'https://www.example.com/test' valid? {is_valid_domain('https://www.example.com/test')}")
    print(f"Is 'https://unapproved.com/test' valid? {is_valid_domain('https://unapproved.com/test')}")
    print(f"Is 'https://www.unapproved.com/test' valid? {is_valid_domain('https://www.unapproved.com/test')}")
