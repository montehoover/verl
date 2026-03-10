from urllib.parse import urlparse, urljoin

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def validate_url(url: str) -> bool:
    """
    Validates a given URL.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid.

    Raises:
        ValueError: If the URL is invalid.
    """
    try:
        result = urlparse(url)
        # A valid URL must have a scheme (e.g., http, https) and a netloc (e.g., www.example.com)
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"Invalid URL: {url}. Scheme and network location are required.")
    except Exception as e: # Catch any parsing errors from urlparse itself or our custom ValueError
        if isinstance(e, ValueError): # Re-raise our custom ValueError
            raise
        # Wrap other exceptions in ValueError for consistent error handling
        raise ValueError(f"Invalid URL: {url}. Reason: {e}")

def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenates a base URL with a path.

    Args:
        base_url: The base URL (e.g., "http://www.example.com").
        path: The path to append (e.g., "/users/profile" or "users/profile").

    Returns:
        The full URL as a string.
    """
    if not validate_url(base_url): # Reuse existing validation
        # validate_url already raises ValueError, so we can just call it.
        # This line is more for explicit control flow, though validate_url would raise on its own.
        pass # Or raise a more specific error if needed, but validate_url's error is good.

    # urljoin handles slashes correctly (e.g., if base_url ends with / and path starts with /)
    return urljoin(base_url, path)

def construct_redirect_link(domain_base_url: str, next_redirect_param: str) -> str:
    """
    Constructs a secure redirect URL.

    The redirect URL must be on the same domain as domain_base_url or on a domain
    listed in ALLOWED_DOMAINS.

    Args:
        domain_base_url: The base URL of the application (e.g., "https://myapp.com").
        next_redirect_param: The user-supplied redirect path or URL
                             (e.g., "/dashboard", "https://partner-domain.org/profile").

    Returns:
        The full, validated redirect URL.

    Raises:
        ValueError: If domain_base_url is invalid, or if the resulting redirect URL
                    is invalid, malformed, or points to an untrusted domain.
    """
    # Validate the base URL first
    if not validate_url(domain_base_url):
        # validate_url already raises ValueError with details
        pass # Should not be reached if validate_url raises as expected

    base_url_parsed = urlparse(domain_base_url)

    # Construct the potential redirect URL using urljoin.
    # urljoin handles various cases:
    # - next_redirect_param is a relative path (e.g., "/foo")
    # - next_redirect_param is an absolute URL (e.g., "http://example.com/foo")
    # - next_redirect_param is a scheme-relative URL (e.g., "//example.com/foo")
    potential_redirect_url = urljoin(domain_base_url, next_redirect_param)

    # Validate the constructed URL (e.g. to catch javascript:, data: schemes, or missing netloc)
    if not validate_url(potential_redirect_url):
        # validate_url will raise ValueError with details
        pass # Should not be reached

    redirect_url_parsed = urlparse(potential_redirect_url)

    # Security check: Ensure the redirect domain is trusted.
    redirect_domain = redirect_url_parsed.netloc
    base_domain = base_url_parsed.netloc

    if redirect_domain == base_domain or redirect_domain in ALLOWED_DOMAINS:
        return potential_redirect_url
    else:
        raise ValueError(
            f"Redirect to untrusted domain '{redirect_domain}' is not allowed. "
            f"Base domain is '{base_domain}', allowed external domains are {ALLOWED_DOMAINS}."
        )

if __name__ == '__main__':
    # Example Usage
    valid_urls = [
        "http://www.example.com",
        "https://example.com/path?query=value#fragment",
        "ftp://user:password@host:port/path",
    ]

    invalid_urls = [
        "www.example.com",  # Missing scheme
        "http://",  # Missing netloc
        "just_a_string",
        None, # type: ignore
        123, # type: ignore
        "http:///path", # Missing netloc, but urlparse might still parse it
        "://example.com" # Missing scheme
    ]

    print("Testing valid URLs:")
    for url_str in valid_urls:
        try:
            if validate_url(url_str):
                print(f"'{url_str}' is valid.")
        except ValueError as e:
            print(f"Error validating '{url_str}': {e}")

    print("\nTesting invalid URLs:")
    for url_str in invalid_urls:
        try:
            validate_url(url_str)
            print(f"'{url_str}' was unexpectedly considered valid.")
        except ValueError as e:
            print(f"Correctly identified '{url_str}' as invalid: {e}")
        except TypeError as e: # Catching type errors for None/int inputs
             print(f"Correctly identified '{url_str}' as invalid (type error): {e}")

    # Test case for urlparse behavior with "http:///path"
    # urlparse("http:///path") -> ParseResult(scheme='http', netloc='', path='/path', params='', query='', fragment='')
    # Our validation should catch this.
    print("\nSpecific test for 'http:///path':")
    try:
        validate_url("http:///path")
        print("'http:///path' was unexpectedly considered valid.")
    except ValueError as e:
        print(f"Correctly identified 'http:///path' as invalid: {e}")

    print("\nTesting URL concatenation:")
    test_cases_concat = [
        ("http://www.example.com", "path/to/resource", "http://www.example.com/path/to/resource"),
        ("http://www.example.com/", "path/to/resource", "http://www.example.com/path/to/resource"),
        ("http://www.example.com", "/path/to/resource", "http://www.example.com/path/to/resource"),
        ("http://www.example.com/", "/path/to/resource", "http://www.example.com/path/to/resource"),
        ("http://www.example.com/api", "v1/users", "http://www.example.com/api/v1/users"),
        ("http://www.example.com/api/", "v1/users", "http://www.example.com/api/v1/users"),
        ("http://www.example.com/api", "/v1/users", "http://www.example.com/api/v1/users"),
        ("http://www.example.com/api/", "/v1/users", "http://www.example.com/api/v1/users"),
        ("https://example.com/test/", "../another", "https://example.com/another"), # Relative path
        ("http://www.example.com", "path with spaces", "http://www.example.com/path%20with%20spaces"), # Path encoding
    ]

    for base, path_segment, expected in test_cases_concat:
        try:
            result = concatenate_url_path(base, path_segment)
            if result == expected:
                print(f"concatenate_url_path('{base}', '{path_segment}') == '{result}' (Correct)")
            else:
                print(f"concatenate_url_path('{base}', '{path_segment}') == '{result}' (Incorrect, expected '{expected}')")
        except ValueError as e:
            print(f"Error concatenating '{base}' and '{path_segment}': {e}")

    print("\nTesting URL concatenation with invalid base URL:")
    try:
        concatenate_url_path("not_a_valid_base", "path")
        print("Concatenation with invalid base URL did not raise ValueError as expected.")
    except ValueError as e:
        print(f"Correctly caught error for invalid base URL: {e}")

    print("\nTesting secure redirect link construction:")
    base_app_url = "https://myapp.com/app"
    test_cases_redirect = [
        # Valid cases
        (base_app_url, "/dashboard", "https://myapp.com/dashboard"),
        (base_app_url, "profile", "https://myapp.com/app/profile"),
        (base_app_url, "https://myapp.com/settings", "https://myapp.com/settings"),
        (base_app_url, "//myapp.com/another", "https://myapp.com/another"),
        (base_app_url, "https://trusted-site.com/user/123", "https://trusted-site.com/user/123"),
        (base_app_url, "//trusted-site.com/api/data", "https://trusted-site.com/api/data"),
        (base_app_url, "https://partner-domain.org", "https://partner-domain.org"),
        ("http://localhost:8000", "/admin", "http://localhost:8000/admin"),

        # Invalid cases - untrusted domains
        (base_app_url, "https://evil-site.com/login", ValueError),
        (base_app_url, "//evil-site.com/path", ValueError),
        (base_app_url, "http://other.myapp.com/path", ValueError), # Subdomain not explicitly base or allowed

        # Invalid cases - malformed redirect param leading to invalid URL by validate_url
        (base_app_url, "javascript:alert(1)", ValueError),
        (base_app_url, "data:text/html,<html> опасный код </html>", ValueError),
        (base_app_url, "http:///path-no-netloc", ValueError),

        # Invalid cases - malformed base_url
        ("myapp.com_no_scheme", "/dashboard", ValueError),
        ("http://", "/dashboard", ValueError),
    ]

    for base, param, expected in test_cases_redirect:
        try:
            result = construct_redirect_link(base, param)
            if expected == ValueError:
                print(f"construct_redirect_link('{base}', '{param}') DID NOT RAISE ValueError, got '{result}'")
            elif result == expected:
                print(f"construct_redirect_link('{base}', '{param}') == '{result}' (Correct)")
            else:
                print(f"construct_redirect_link('{base}', '{param}') == '{result}' (Incorrect, expected '{expected}')")
        except ValueError as e:
            if expected == ValueError:
                print(f"construct_redirect_link('{base}', '{param}') correctly raised ValueError: {e}")
            else:
                print(f"construct_redirect_link('{base}', '{param}') UNEXPECTEDLY raised ValueError: {e}")
        except Exception as e:
            print(f"construct_redirect_link('{base}', '{param}') UNEXPECTEDLY raised {type(e).__name__}: {e}")
