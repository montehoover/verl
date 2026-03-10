from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def extract_url_components(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary with the URL components (scheme, domain, path).
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "domain": parsed_url.netloc,
        "path": parsed_url.path,
    }

def check_url_validity(url_string, allowed_domains):
    """
    Verifies if the URL's scheme is 'http' or 'https' and if the domain
    is in a predefined list of allowed domains.

    Args:
        url_string: The URL string to validate.
        allowed_domains: A list of allowed domain names.

    Returns:
        True if the URL is valid, False otherwise.
    """
    parsed_url = urlparse(url_string)

    if parsed_url.scheme not in ['http', 'https']:
        return False

    if parsed_url.netloc not in allowed_domains:
        return False

    return True

def resolve_image_url(image_url):
    """
    Ensures an image URL belongs to a trusted image hosting service and has an allowed scheme.

    Args:
        image_url: The image URL string.

    Returns:
        The resolved URL if it is from a trusted domain and has an allowed scheme.

    Raises:
        ValueError: If the URL's scheme is not allowed or domain is not trusted.
    """
    parsed_url = urlparse(image_url)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Allowed schemes are: {ALLOWED_SCHEMES}")

    if parsed_url.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted image host: {parsed_url.netloc}. Allowed hosts are: {TRUSTED_IMAGE_HOSTS}")

    return image_url

if __name__ == '__main__':
    # Example usage for extract_url_components:
    test_url = "https://www.example.com/path/to/resource?query=param#fragment"
    components = extract_url_components(test_url)
    print(f"URL: {test_url}")
    print(f"Scheme: {components['scheme']}")
    print(f"Domain: {components['domain']}")
    print(f"Path: {components['path']}")

    test_url_2 = "http://localhost:8080/api/v1/users?id=123"
    components_2 = extract_url_components(test_url_2)
    print(f"\nURL: {test_url_2}")
    print(f"Scheme: {components_2['scheme']}")
    print(f"Domain: {components_2['domain']}")
    print(f"Path: {components_2['path']}")

    test_url_3 = "ftp://user:password@ftp.example.com/files/archive.zip"
    components_3 = extract_url_components(test_url_3)
    print(f"\nURL: {test_url_3}")
    print(f"Scheme: {components_3['scheme']}")
    print(f"Domain: {components_3['domain']}")
    print(f"Path: {components_3['path']}")

    # Example usage for check_url_validity:
    allowed_domains_list = ["www.example.com", "api.example.org", "localhost:8080"]

    url_to_check_1 = "https://www.example.com/path/to/page"
    is_valid_1 = check_url_validity(url_to_check_1, allowed_domains_list)
    print(f"\nURL: {url_to_check_1}, Allowed Domains: {allowed_domains_list}, Valid: {is_valid_1}")

    url_to_check_2 = "http://api.example.org/data"
    is_valid_2 = check_url_validity(url_to_check_2, allowed_domains_list)
    print(f"URL: {url_to_check_2}, Allowed Domains: {allowed_domains_list}, Valid: {is_valid_2}")

    url_to_check_3 = "ftp://www.example.com/files" # Invalid scheme
    is_valid_3 = check_url_validity(url_to_check_3, allowed_domains_list)
    print(f"URL: {url_to_check_3}, Allowed Domains: {allowed_domains_list}, Valid: {is_valid_3}")

    url_to_check_4 = "https://www.anotherdomain.com/resource" # Domain not allowed
    is_valid_4 = check_url_validity(url_to_check_4, allowed_domains_list)
    print(f"URL: {url_to_check_4}, Allowed Domains: {allowed_domains_list}, Valid: {is_valid_4}")

    url_to_check_5 = "http://localhost:8080/test"
    is_valid_5 = check_url_validity(url_to_check_5, allowed_domains_list)
    print(f"URL: {url_to_check_5}, Allowed Domains: {allowed_domains_list}, Valid: {is_valid_5}")

    # Example usage for resolve_image_url:
    print("\n--- Testing resolve_image_url ---")
    trusted_url_1 = "https://img.example.com/image.jpg"
    try:
        resolved_url = resolve_image_url(trusted_url_1)
        print(f"Original URL: {trusted_url_1}, Resolved URL: {resolved_url} (Valid)")
    except ValueError as e:
        print(f"Original URL: {trusted_url_1}, Error: {e}")

    trusted_url_2 = "http://cdn.trusted.com/path/to/image.png"
    try:
        resolved_url = resolve_image_url(trusted_url_2)
        print(f"Original URL: {trusted_url_2}, Resolved URL: {resolved_url} (Valid)")
    except ValueError as e:
        print(f"Original URL: {trusted_url_2}, Error: {e}")

    untrusted_domain_url = "https://other.domain.com/pic.gif"
    try:
        resolved_url = resolve_image_url(untrusted_domain_url)
        print(f"Original URL: {untrusted_domain_url}, Resolved URL: {resolved_url} (Valid)")
    except ValueError as e:
        print(f"Original URL: {untrusted_domain_url}, Error: {e}")

    invalid_scheme_url = "ftp://img.example.com/image.bmp"
    try:
        resolved_url = resolve_image_url(invalid_scheme_url)
        print(f"Original URL: {invalid_scheme_url}, Resolved URL: {resolved_url} (Valid)")
    except ValueError as e:
        print(f"Original URL: {invalid_scheme_url}, Error: {e}")

    untrusted_and_invalid_scheme = "gopher://untrusted.host/img.raw"
    try:
        resolved_url = resolve_image_url(untrusted_and_invalid_scheme)
        print(f"Original URL: {untrusted_and_invalid_scheme}, Resolved URL: {resolved_url} (Valid)")
    except ValueError as e:
        print(f"Original URL: {untrusted_and_invalid_scheme}, Error: {e}")
