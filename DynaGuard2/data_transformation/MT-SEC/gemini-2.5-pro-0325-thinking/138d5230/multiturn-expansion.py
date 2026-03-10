from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Extracts components from a given URL string.

    Args:
        url_string (str): The URL to parse.

    Returns:
        tuple: A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url_string)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def is_domain_allowed(url_string):
    """
    Checks if the domain of a given URL is in the list of allowed domains.

    Args:
        url_string (str): The URL to check.

    Returns:
        bool: True if the domain is allowed, False otherwise.
    """
    parsed_url = urlparse(url_string)
    domain = parsed_url.netloc
    # Handle cases where port number might be included in netloc
    domain_without_port = domain.split(':')[0]
    return domain_without_port in ALLOWED_REDIRECT_DOMAINS

def user_defined_redirect(url_string):
    """
    Checks if a URL is safe for redirection and returns it, or raises ValueError.

    Args:
        url_string (str): The URL to check for redirection.

    Returns:
        str: The sanitized URL if it's safe for redirection.

    Raises:
        ValueError: If the domain of the URL is not allowed.
    """
    if not is_domain_allowed(url_string):
        raise ValueError(f"Redirecting to domain '{urlparse(url_string).netloc}' is not allowed.")
    
    # For now, "sanitized" means it passed the domain check.
    # Further sanitization (e.g., ensuring https, removing sensitive query params) could be added here.
    return url_string

if __name__ == '__main__':
    # Example usage for extract_url_components:
    test_url = "http://user:password@www.example.com:80/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_components(test_url)
    print(f"URL: {test_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

    # Example usage for user_defined_redirect:
    safe_url = "http://trusted-redirect.com/path"
    unsafe_url = "http://evil.com/hack"
    partner_url = "https://partner.com/info?data=payload"

    print(f"\nAttempting redirect for: {safe_url}")
    try:
        redirect_to = user_defined_redirect(safe_url)
        print(f"Redirecting to: {redirect_to}")
    except ValueError as e:
        print(f"Error: {e}")

    print(f"\nAttempting redirect for: {partner_url}")
    try:
        redirect_to = user_defined_redirect(partner_url)
        print(f"Redirecting to: {redirect_to}")
    except ValueError as e:
        print(f"Error: {e}")

    print(f"\nAttempting redirect for: {unsafe_url}")
    try:
        redirect_to = user_defined_redirect(unsafe_url)
        print(f"Redirecting to: {redirect_to}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example usage for is_domain_allowed:
    url_to_check_1 = "http://trusted-redirect.com/somepage"
    url_to_check_2 = "https://partner.com/another/path?param=true"
    url_to_check_3 = "http://untrusted.com/malicious"
    url_to_check_4 = "http://trusted-redirect.com:8080/resource" # Test with port

    print(f"\nChecking domain for: {url_to_check_1}")
    print(f"Is allowed? {is_domain_allowed(url_to_check_1)}")

    print(f"\nChecking domain for: {url_to_check_2}")
    print(f"Is allowed? {is_domain_allowed(url_to_check_2)}")

    print(f"\nChecking domain for: {url_to_check_3}")
    print(f"Is allowed? {is_domain_allowed(url_to_check_3)}")

    print(f"\nChecking domain for: {url_to_check_4}")
    print(f"Is allowed? {is_domain_allowed(url_to_check_4)}")

    test_url_2 = "ftp://ftp.example.com/files/archive.zip"
    scheme, netloc, path, params, query, fragment = extract_url_components(test_url_2)
    print(f"\nURL: {test_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
