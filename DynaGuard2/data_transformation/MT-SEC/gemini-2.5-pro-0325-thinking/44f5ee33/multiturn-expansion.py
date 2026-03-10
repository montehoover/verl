from urllib.parse import urlparse
import urllib.parse # Add this import

ALLOWED_DOMAINS = ["example.com", "example.org", "test.com"]

def validate_url_domain(url: str):
    """
    Validates if the domain of the given URL is in the ALLOWED_DOMAINS list.

    Args:
        url: The URL string to validate.

    Raises:
        ValueError: If the URL's domain is not in ALLOWED_DOMAINS 
                    or if the URL is malformed.
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Common case: remove www. prefix if present for comparison
        if domain.startswith("www."):
            domain = domain[4:]
        
        if not domain:
            raise ValueError(f"Could not parse domain from URL: {url}")

        if domain not in ALLOWED_DOMAINS:
            raise ValueError(f"Domain '{domain}' is not an allowed domain.")
    except Exception as e:
        # Catch potential errors from urlparse itself or re-raise our ValueErrors
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid URL format: {url}") from e

def combine_base_with_path(base: str, path: str) -> str:
    """
    Combines a base URL with a path, handling leading/trailing slashes.

    Args:
        base: The base URL string.
        path: The path string to append.

    Returns:
        The full combined URL string.
    """
    if base.endswith('/'):
        base = base[:-1]
    if path.startswith('/'):
        path = path[1:]
    return f"{base}/{path}"

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def build_oauth_redirect_url(main_url: str, redirect_path: str, nonce: str) -> str:
    """
    Builds an OAuth callback URL, validates its domain, and appends a nonce.

    Args:
        main_url: The main URL for the callback (e.g., "https://auth.myapp.com").
        redirect_path: The path for the redirect (e.g., "/callback").
        nonce: A unique string to be included as a query parameter for security.

    Returns:
        The full OAuth callback URL string.

    Raises:
        ValueError: If the domain of main_url is not in ALLOWED_CALLBACK_DOMAINS
                    or if the URL is malformed.
    """
    try:
        parsed_main_url = urlparse(main_url)
        domain = parsed_main_url.netloc
        
        if not domain:
            raise ValueError(f"Could not parse domain from main_url: {main_url}")

        if domain not in ALLOWED_CALLBACK_DOMAINS:
            raise ValueError(f"Domain '{domain}' is not an allowed callback domain.")

        # Combine base URL and path
        # Ensure main_url doesn't have a path component that gets overridden
        # and redirect_path is correctly appended.
        # We can use a similar logic to combine_base_with_path or urljoin
        
        base_url_for_combine = f"{parsed_main_url.scheme}://{parsed_main_url.netloc}"
        full_path = combine_base_with_path(base_url_for_combine, redirect_path)

        # Add nonce as a query parameter
        # urlparse and urlunparse can handle query parameters robustly
        parsed_full_path = urlparse(full_path)
        query_params = urllib.parse.parse_qs(parsed_full_path.query)
        query_params['nonce'] = [nonce] # parse_qs returns list values
        
        # Rebuild the URL with the new query string
        # Ensure path is correctly handled if it was empty initially
        final_url = urllib.parse.urlunparse((
            parsed_full_path.scheme,
            parsed_full_path.netloc,
            parsed_full_path.path if parsed_full_path.path else '/', # Ensure path is at least '/'
            parsed_full_path.params,
            urllib.parse.urlencode(query_params, doseq=True),
            parsed_full_path.fragment
        ))
        return final_url

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error building OAuth redirect URL: {e}") from e

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/page",
        "https://www.example.org/another?query=param",
        "ftp://test.com/resource",
        "http://sub.example.com/path", # This will fail as "sub.example.com" is not in ALLOWED_DOMAINS
        "http://invalid-domain.net/page",
        "not_a_url",
        "http://example.com", # Test without path
        "https://www.test.com", # Test with www and without path
    ]

    for url in test_urls:
        try:
            validate_url_domain(url)
            print(f"URL '{url}' is valid.")
        except ValueError as e:
            print(f"Validation failed for URL '{url}': {e}")

    print("\nTesting with a different set of allowed domains:")
    # Example of how to change allowed domains if needed (though typically it's a global constant)
    original_allowed_domains = list(ALLOWED_DOMAINS) # make a copy
    ALLOWED_DOMAINS.clear()
    ALLOWED_DOMAINS.extend(["another.com", "specific.io"])
    print(f"New allowed domains: {ALLOWED_DOMAINS}")

    new_test_urls = [
        "http://another.com/path",
        "https://www.specific.io/resource",
        "http://example.com/page" # This should fail now
    ]
    for url in new_test_urls:
        try:
            validate_url_domain(url)
            print(f"URL '{url}' is valid.")
        except ValueError as e:
            print(f"Validation failed for URL '{url}': {e}")
    
    # Restore original for other potential tests if this were a module
    ALLOWED_DOMAINS.clear()
    ALLOWED_DOMAINS.extend(original_allowed_domains)

    print("\nTesting combine_base_with_path function:")
    test_combinations = [
        ("http://example.com", "path/to/resource"),
        ("http://example.com/", "path/to/resource"),
        ("http://example.com", "/path/to/resource"),
        ("http://example.com/", "/path/to/resource"),
        ("http://example.com/api", "v1/users"),
        ("http://example.com/api/", "/v1/users/"), # Path's trailing slash should be preserved if intended
    ]
    for base, path_segment in test_combinations:
        combined_url = combine_base_with_path(base, path_segment)
        print(f"Base: '{base}', Path: '{path_segment}' -> Combined: '{combined_url}'")

    print("\nTesting build_oauth_redirect_url function:")
    oauth_test_cases = [
        ("https://auth.myapp.com", "/callback", "securenonce123"),
        ("http://login.myapp.org", "oauth/receive", "anothernonce456"), # HTTP should be fine if domain is allowed
        ("https://oauth.myapp.net/v2", "/redirect", "yetanother789"), # main_url with path
        ("https://unauthorized.com", "/callback", "failnonce000"), # Should fail
        ("not_a_valid_url", "/callback", "invalidurlnonce"), # Should fail
        ("https://auth.myapp.com", "/callback?existing_param=true", "noncewithparams"), # Path with existing params
    ]

    for main_url, path, nonce in oauth_test_cases:
        try:
            redirect_url = build_oauth_redirect_url(main_url, path, nonce)
            print(f"OAuth Redirect URL for '{main_url}', '{path}', '{nonce}': {redirect_url}")
        except ValueError as e:
            print(f"Failed for '{main_url}', '{path}', '{nonce}': {e}")
