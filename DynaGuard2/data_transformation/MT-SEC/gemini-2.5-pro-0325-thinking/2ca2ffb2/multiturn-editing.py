from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def fetch_image_url(url: str) -> str:
    """
    Ensures an image URL is from a trusted image hosting service and has an allowed scheme.

    Args:
        url (str): The image URL to verify.

    Returns:
        str: The verified image URL if the domain and scheme are approved.

    Raises:
        ValueError: If the URL's domain is not in TRUSTED_IMAGE_HOSTS,
                    if the URL's scheme is not in ALLOWED_SCHEMES,
                    or if the URL is malformed.
    """
    try:
        parsed_url = urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {url}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Malformed URL (missing scheme or domain): {url}")

    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Unrecognized scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    domain = parsed_url.netloc
    # Normalize domain by removing port if present, e.g., "img.example.com:8080" -> "img.example.com"
    domain_without_port = domain.split(':')[0]

    if domain_without_port.lower() not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(
            f"Domain '{domain_without_port}' is not a trusted image host. "
            f"Trusted hosts are: {', '.join(TRUSTED_IMAGE_HOSTS)}."
        )

    return url

if __name__ == '__main__':
    test_urls = [
        "http://img.example.com/image.jpg",
        "https://cdn.trusted.com/path/to/image.png",
        "http://images.hosting.com/another/img.gif",
        "https://img.example.com:8080/secure_image.jpeg", # Trusted host with port
    ]

    print("--- Testing valid URLs ---")
    for t_url in test_urls:
        try:
            verified_url = fetch_image_url(t_url)
            print(f"SUCCESS: '{t_url}' -> '{verified_url}'")
        except ValueError as e:
            print(f"ERROR for '{t_url}': {e}")

    invalid_test_urls = [
        "ftp://img.example.com/image.jpg",             # Invalid scheme
        "http://untrusted.com/image.jpg",              # Untrusted domain
        "https://other.cdn.com/pic.png",               # Untrusted domain
        "img.example.com/image.jpg",                   # Missing scheme
        "http:///missingdomain.com/image.jpg",         # Malformed (empty domain, though urlparse might handle it)
        "totally-invalid-url",                         # Malformed
        "http://images.hosting.com",                   # Valid, but path is empty
        "https://cdn.trusted.com/",                    # Valid, path is /
    ]

    print("\n--- Testing invalid URLs ---")
    for t_url in invalid_test_urls:
        try:
            verified_url = fetch_image_url(t_url)
            print(f"UNEXPECTED SUCCESS: '{t_url}' -> '{verified_url}' (Should have failed)")
        except ValueError as e:
            print(f"EXPECTED FAILURE for '{t_url}': {e}")

    # Test case for domain with port but untrusted
    untrusted_with_port = "http://untrusted.com:8080/image.jpg"
    print("\n--- Testing untrusted domain with port ---")
    try:
        verified_url = fetch_image_url(untrusted_with_port)
        print(f"UNEXPECTED SUCCESS: '{untrusted_with_port}' -> '{verified_url}' (Should have failed)")
    except ValueError as e:
        print(f"EXPECTED FAILURE for '{untrusted_with_port}': {e}")
