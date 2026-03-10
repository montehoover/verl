from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def construct_redirect_link(domain_base_url: str, next_redirect_param: str) -> str:
    """
    Constructs a redirect URL and validates it against the base domain or allowed external domains.

    Args:
        domain_base_url: The base URL of the application (e.g., "https://app.example.com").
        next_redirect_param: The URL path or full address for redirection
                             (e.g., "/dashboard", "https://trusted-site.com/profile").

    Returns:
        The final constructed redirect URL as a string.

    Raises:
        TypeError: If inputs are not of the expected type.
        ValueError: If domain_base_url is invalid, or if the generated redirect URL
                    is not within the base domain or not part of ALLOWED_DOMAINS.
    """
    if not isinstance(domain_base_url, str):
        raise TypeError("domain_base_url must be a string.")
    if not isinstance(next_redirect_param, str):
        raise TypeError("next_redirect_param must be a string.")

    if not domain_base_url:
        raise ValueError("domain_base_url cannot be empty.")

    parsed_base = urlparse.urlparse(domain_base_url)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise ValueError(f"Invalid domain_base_url: '{domain_base_url}'. It must be a full URL with scheme and domain.")
    
    # Extract netloc (domain) without port for comparison
    base_netloc_no_port = parsed_base.netloc.split(':')[0]

    # Determine if next_redirect_param is a full URL or a path
    parsed_next = urlparse.urlparse(next_redirect_param)

    if parsed_next.scheme and parsed_next.netloc:
        # next_redirect_param is an absolute URL
        final_redirect_url = next_redirect_param
    else:
        # next_redirect_param is a relative path or an absolute path component (e.g. /foo).
        # Ensure domain_base_url is treated as a base "directory" for urljoin.
        effective_base = domain_base_url
        if not effective_base.endswith('/'):
            effective_base += '/'
        final_redirect_url = urlparse.urljoin(effective_base, next_redirect_param)

    # Validate the final_redirect_url
    parsed_final_redirect = urlparse.urlparse(final_redirect_url)
    if not parsed_final_redirect.scheme or not parsed_final_redirect.netloc:
        raise ValueError(f"Generated redirect URL '{final_redirect_url}' is malformed or has no domain.")

    redirect_netloc_no_port = parsed_final_redirect.netloc.split(':')[0]

    if redirect_netloc_no_port == base_netloc_no_port or redirect_netloc_no_port in ALLOWED_DOMAINS:
        return final_redirect_url
    else:
        raise ValueError(
            f"Redirect URL '{final_redirect_url}' is not allowed. "
            f"Domain '{redirect_netloc_no_port}' is not the application's base domain ('{base_netloc_no_port}') "
            f"and not in the allowed external domains: {ALLOWED_DOMAINS}."
        )

if __name__ == '__main__':
    print("--- Testing construct_redirect_link ---")
    
    base_app_url = "https://app.example.com"
    base_app_url_with_path = "https://app.example.com/some/app"
    base_app_url_with_port = "https://app.example.com:8080"

    # (domain_base_url, next_redirect_param, expected_output_or_ErrorType)
    test_cases = [
        # Valid: Relative path, same domain
        (base_app_url, "/dashboard", "https://app.example.com/dashboard"),
        (base_app_url, "user/profile", "https://app.example.com/user/profile"),
        (base_app_url + "/", "user/profile", "https://app.example.com/user/profile"),
        (base_app_url_with_path, "settings", "https://app.example.com/some/app/settings"),
        (base_app_url_with_path, "/overview", "https://app.example.com/overview"), # Absolute path
        (base_app_url_with_port, "/settings", "https://app.example.com:8080/settings"),

        # Valid: Absolute URL, same domain
        (base_app_url, "https://app.example.com/another/page", "https://app.example.com/another/page"),
        (base_app_url, "http://app.example.com/page", "http://app.example.com/page"), # Scheme change, same domain
        (base_app_url_with_port, "https://app.example.com:8080/another", "https://app.example.com:8080/another"),
        (base_app_url, "https://app.example.com:9000/another", "https://app.example.com:9000/another"), # Different port, same domain

        # Valid: Absolute URL, allowed external domain
        (base_app_url, "https://trusted-site.com", "https://trusted-site.com"),
        (base_app_url, "http://partner-domain.org/some/path?query=1", "http://partner-domain.org/some/path?query=1"),
        (base_app_url, "https://trusted-site.com:8000/path", "https://trusted-site.com:8000/path"),

        # Invalid: Absolute URL, disallowed external domain
        (base_app_url, "https://malicious-site.com", ValueError),
        (base_app_url, "http://google.com", ValueError),
        (base_app_url, "https://sub.app.example.com", ValueError), # Subdomain not implicitly allowed

        # Invalid: Malformed domain_base_url
        ("app.example.com", "/dashboard", ValueError), # No scheme
        ("http://", "/dashboard", ValueError), # No domain
        
        # Valid: FTP scheme for domain_base_url (domain validation still applies)
        ("ftp://app.example.com/data", "/dashboard", "ftp://app.example.com/dashboard"),

        # Empty or tricky next_redirect_param
        (base_app_url, "", base_app_url + "/"), # Empty path resolves to base + "/"
        (base_app_url_with_path, "", base_app_url_with_path + "/"),
        (base_app_url, "//otherdomain.com/path", ValueError), # Protocol-relative URL to disallowed domain
        (base_app_url, "//trusted-site.com/path", "https://trusted-site.com/path"), # Protocol-relative to allowed, assumes https
        (base_app_url, "javascript:alert(1)", ValueError), # Invalid scheme in redirect

        # Type errors
        (123, "/dashboard", TypeError),
        (base_app_url, 123, TypeError),
        (None, "/dashboard", TypeError),
        (base_app_url, None, TypeError),
        (base_app_url, {"path": "test"}, TypeError),
    ]

    for i, (dbu, nrp, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: construct_redirect_link({repr(dbu)}, {repr(nrp)})")
        try:
            result = construct_redirect_link(dbu, nrp)
            print(f"  Result:   '{result}'")
            if isinstance(expected, type) and issubclass(expected, Exception):
                print(f"  FAIL: Expected Error {expected.__name__}, but got result '{result}'")
            elif result != expected:
                 print(f"  FAIL: Expected '{expected}', but got '{result}'")
            else:
                print("  PASS")
        except (ValueError, TypeError) as e:
            print(f"  Error:    {e.__class__.__name__}: {e}")
            if not isinstance(e, expected):
                print(f"  FAIL: Expected Error {expected.__name__ if isinstance(expected, type) else 'specific value'}, but got {e.__class__.__name__}")
            else:
                print("  PASS")
        except Exception as e: # Catch any other unexpected errors
            print(f"  UNEXPECTED Error: {e.__class__.__name__}: {e}")
            print(f"  FAIL: Expected {expected}, but got unexpected {e.__class__.__name__}")
