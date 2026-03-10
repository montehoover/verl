from urllib import parse as urlparse_lib # Using alias as requested

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def get_normalized_domain(url_string: str) -> str:
    """
    Parses a URL string and returns its normalized domain (e.g., "example.com" from "http://www.example.com").
    Returns an empty string if the URL is malformed, has no domain, or an error occurs.
    """
    if not url_string:
        return ""
    try:
        parsed = urlparse_lib.urlparse(url_string)
        domain = parsed.netloc
        if not domain:
            return ""
        if domain.startswith('www.'):
            return domain[4:]
        return domain
    except Exception: # Catch any parsing errors (e.g., from invalid characters in URL)
        return ""

def generate_redirect_link(app_url: str, redirect_param: str) -> str:
    """
    Generates a final redirect URL by combining app_url and redirect_param.
    The domain of the final URL must either match the domain of app_url
    or be present in the APPROVED_DOMAINS list.
    The function normalizes domains by removing "www." before comparison.

    Args:
        app_url: The base URL of the application (e.g., "https://myapp.com").
        redirect_param: The path or full URL for redirection (e.g., "/dashboard" or "https://partner.com/page").

    Returns:
        The final redirect URL as a string if valid.

    Raises:
        ValueError: If app_url is invalid, if the final URL's domain is not allowed,
                    or if the final URL is malformed or uses a non-HTTP/HTTPS scheme.
    """
    if not app_url:
        raise ValueError("app_url cannot be empty.")

    parsed_app_url_check = urlparse_lib.urlparse(app_url)
    if not parsed_app_url_check.scheme or not parsed_app_url_check.netloc:
        raise ValueError(f"app_url '{app_url}' must be a valid absolute URL (e.g., 'http://example.com').")

    app_normalized_domain = get_normalized_domain(app_url)
    # This should not be empty due to the check above, but as a safeguard:
    if not app_normalized_domain:
         raise ValueError(f"Could not parse a valid domain from app_url: {app_url}")

    # Construct the final redirect URL using urljoin.
    # urljoin correctly handles redirect_param being a full URL, absolute path, or relative path.
    final_redirect_url = urlparse_lib.urljoin(app_url, redirect_param)

    # Validate the scheme of the final URL
    parsed_final_url = urlparse_lib.urlparse(final_redirect_url)
    if parsed_final_url.scheme.lower() not in ('http', 'https'):
        raise ValueError(f"Redirect URL scheme '{parsed_final_url.scheme}' is not allowed. Only http/https are permitted.")

    # Get and validate the normalized domain of the final redirect URL
    final_redirect_normalized_domain = get_normalized_domain(final_redirect_url)
    if not final_redirect_normalized_domain:
        raise ValueError(f"Final redirect URL '{final_redirect_url}' does not have a valid domain or is malformed.")

    # Check if the final domain is allowed
    is_app_domain = (final_redirect_normalized_domain == app_normalized_domain)
    # APPROVED_DOMAINS are expected to be stored without "www."
    is_approved_external_domain = (final_redirect_normalized_domain in APPROVED_DOMAINS)

    if is_app_domain or is_approved_external_domain:
        return final_redirect_url
    else:
        raise ValueError(
            f"Redirect to domain '{final_redirect_normalized_domain}' is not allowed. "
            f"Application domain: '{app_normalized_domain}'. Approved external domains: {APPROVED_DOMAINS}."
        )

if __name__ == '__main__':
    print("Testing generate_redirect_link function:\n")

    test_cases = [
        # app_url, redirect_param, expected_success, expected_url (if success)
        ("https://myapp.com", "/dashboard", True, "https://myapp.com/dashboard"),
        ("http://www.myapp.com", "/profile", True, "http://www.myapp.com/profile"),
        ("https://myapp.com", "settings", True, "https://myapp.com/settings"),
        ("https://myapp.com/path/", "another", True, "https://myapp.com/path/another"),
        ("https://myapp.com", "https://trusted-site.com/partner-page", True, "https://trusted-site.com/partner-page"),
        ("https://myapp.com", "http://www.trusted-site.com/other", True, "http://www.trusted-site.com/other"),
        ("https://myapp.com", "https://partner-domain.org", True, "https://partner-domain.org"),
        ("https://myapp.com", "https://untrusted.com/malicious", False, None),
        ("https://myapp.com", "//untrusted.com/schemerel", False, None), # Scheme relative, becomes https://untrusted.com
        ("https://myapp.com", "https://myapp.com/another/path", True, "https://myapp.com/another/path"),
        ("http://www.myapp.com", "https://myapp.com/secure", True, "https://myapp.com/secure"), # Domain matches after normalization
        ("https://myapp.com", "", True, "https://myapp.com/"), # Empty redirect_param
        ("https://myapp.com", "javascript:alert(1)", False, None), # Invalid scheme
        ("ftp://myapp.com", "/dashboard", False, None), # Invalid app_url scheme (though initial check catches this)
        ("myapp.com-no-scheme", "/dashboard", False, None), # Invalid app_url
        ("", "/dashboard", False, None), # Empty app_url
        ("https://myapp.com", "http://evil.com", False, None),
        ("https://www.myapp.com", "https://trusted-site.com", True, "https://trusted-site.com"),
        # Test case where redirect_param is a full URL to the same app but different www
        ("https://myapp.com", "https://www.myapp.com/path", True, "https://www.myapp.com/path"),
        ("https://www.myapp.com", "https://myapp.com/path", True, "https://myapp.com/path"),
    ]

    for i, (app_url, redirect_param, expect_success, expected_url) in enumerate(test_cases):
        print(f"Test Case {i+1}: app_url='{app_url}', redirect_param='{redirect_param}'")
        try:
            result_url = generate_redirect_link(app_url, redirect_param)
            if expect_success:
                print(f"  Success (Expected): Generated URL: {result_url}")
                if result_url != expected_url:
                    print(f"  ERROR: Expected URL '{expected_url}' but got '{result_url}'")
            else:
                print(f"  ERROR (Expected Failure, but Succeeded): Generated URL: {result_url}")
        except ValueError as e:
            if expect_success:
                print(f"  ERROR (Expected Success, but Failed): {e}")
            else:
                print(f"  Success (Expected Failure): Caught ValueError: {e}")
        print("-" * 30)
