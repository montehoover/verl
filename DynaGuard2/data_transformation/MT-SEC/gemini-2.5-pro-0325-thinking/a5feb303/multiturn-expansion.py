from urllib.parse import urlparse

def extract_url_components(url: str) -> tuple[str, str, str, str, str, str]:
    """
    Extracts components from a given URL.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def is_domain_allowed(url: str) -> bool:
    """
    Checks if the domain of the given URL is in the list of allowed domains.

    Args:
        url: The URL string to check.

    Returns:
        True if the domain is allowed, False otherwise.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    # Handle potential port numbers in netloc
    if ":" in domain:
        domain = domain.split(":")[0]
    return domain in ALLOWED_REDIRECT_DOMAINS

def process_user_redirect(url: str) -> str:
    """
    Processes a user-provided URL for redirection.
    Ensures the URL's domain is in the allowed list.

    Args:
        url: The URL string to process.

    Returns:
        The sanitized URL if it's safe for redirection.

    Raises:
        ValueError: If the domain is not allowed.
    """
    if not is_domain_allowed(url):
        raise ValueError(f"Redirect to domain '{urlparse(url).netloc}' is not allowed.")
    # For now, "sanitized" means it passed the domain check.
    # Further sanitization (e.g., ensuring https, removing sensitive query params) could be added here.
    return url

if __name__ == '__main__':
    # Example usage for extract_url_components:
    example_url = "http://user:password@www.example.com:80/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_components(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

    # Example usage for process_user_redirect:
    safe_url = "https://partner.com/secure-redirect"
    unsafe_url = "http://malicious-site.com/phishing"

    print(f"\nProcessing redirect for '{safe_url}':")
    try:
        processed_url = process_user_redirect(safe_url)
        print(f"  Redirect allowed: {processed_url}")
    except ValueError as e:
        print(f"  Error: {e}")

    print(f"\nProcessing redirect for '{unsafe_url}':")
    try:
        processed_url = process_user_redirect(unsafe_url)
        print(f"  Redirect allowed: {processed_url}")
    except ValueError as e:
        print(f"  Error: {e}")

    # Example usage for is_domain_allowed:
    url_to_check_1 = "http://trusted-redirect.com/some/path"
    print(f"\nIs '{url_to_check_1}' allowed? {is_domain_allowed(url_to_check_1)}")

    url_to_check_2 = "https://partner.com:8080/another/path?query=1"
    print(f"Is '{url_to_check_2}' allowed? {is_domain_allowed(url_to_check_2)}")

    url_to_check_3 = "http://untrusted.com/path"
    print(f"Is '{url_to_check_3}' allowed? {is_domain_allowed(url_to_check_3)}")

    url_to_check_4 = "ftp://trusted-redirect.com/resource" # Scheme doesn't matter for domain check
    print(f"Is '{url_to_check_4}' allowed? {is_domain_allowed(url_to_check_4)}")

    example_url_2 = "https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlparse"
    scheme, netloc, path, params, query, fragment = extract_url_components(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
