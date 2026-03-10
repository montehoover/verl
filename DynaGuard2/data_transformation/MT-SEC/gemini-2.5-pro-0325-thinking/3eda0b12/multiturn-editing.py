from urllib import parse as urlparse_module # Renamed to avoid conflict with urlparse function if any

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def get_effective_domain(url_string: str) -> str:
    """Helper to parse a URL and return its effective domain (stripping www.)."""
    parsed_url = urlparse_module.urlparse(url_string)
    domain = parsed_url.netloc
    if not domain:
        raise ValueError(f"Could not extract domain from URL: '{url_string}'")
    if domain.startswith('www.'):
        return domain[4:]
    return domain

def construct_redirect_url(main_url: str, target_param: str) -> str:
    """
    Constructs a redirect URL after a successful login, ensuring it's within
    the application's domain or a permitted external domain.

    Args:
        main_url: The base URL of the application (e.g., "https://app.example.com").
        target_param: The URL path (e.g., "/dashboard") or full URL 
                      (e.g., "https://app.example.com/profile" or "https://trusted-site.com/landing")
                      the user intends to navigate to.

    Returns:
        The final redirect URL as a string.

    Raises:
        ValueError: If main_url is invalid, or if the final redirect URL's domain 
                    is not the same as main_url's domain or not in PERMITTED_DOMAINS.
    """
    try:
        parsed_main_url = urlparse_module.urlparse(main_url)
        if not parsed_main_url.scheme or not parsed_main_url.netloc:
            raise ValueError(f"Invalid main_url: '{main_url}'. It must be a full URL with scheme and domain.")
        main_app_domain_effective = get_effective_domain(main_url)
    except ValueError as e: # Catch error from get_effective_domain if main_url is bad
        raise ValueError(f"Invalid main_url: '{main_url}'. Error: {e}")


    # Determine if target_param is a full URL or a path
    parsed_target = urlparse_module.urlparse(target_param)

    if parsed_target.scheme and parsed_target.netloc:
        # target_param is a full URL
        final_url = target_param
    else:
        # target_param is a path, join it with main_url
        # Ensure main_url ends with a slash if it's just a domain, for urljoin to work as expected for paths.
        # However, urljoin handles this correctly in most cases.
        final_url = urlparse_module.urljoin(main_url, target_param)

    # Validate the domain of the final_url
    try:
        final_url_domain_effective = get_effective_domain(final_url)
    except ValueError as e: # Catch error from get_effective_domain if final_url is bad
        raise ValueError(f"Could not determine domain of the constructed redirect URL '{final_url}'. Error: {e}")


    is_same_domain_as_main_app = (final_url_domain_effective == main_app_domain_effective)
    is_permitted_external_domain = final_url_domain_effective in PERMITTED_DOMAINS

    if not (is_same_domain_as_main_app or is_permitted_external_domain):
        raise ValueError(
            f"Redirect to domain '{parsed_target.netloc or final_url_domain_effective}' "
            f"(from target_param '{target_param}') is not allowed. "
            f"Must be within '{main_app_domain_effective}' or {PERMITTED_DOMAINS}."
        )

    return final_url

if __name__ == '__main__':
    app_main_url = "https://app.example.com"
    app_main_url_www = "https://www.app.example.com"


    test_cases = [
        # Valid cases: Same domain
        (app_main_url, "/dashboard", "Path within main app domain"),
        (app_main_url, "profile/settings", "Path (no leading slash) within main app domain"),
        (app_main_url, "https://app.example.com/user/123", "Full URL within main app domain"),
        (app_main_url, "https://www.app.example.com/user/123", "Full URL (www) within main app domain"),
        (app_main_url_www, "/dashboard", "Path within main app domain (www)"),
        (app_main_url_www, "https://app.example.com/user/123", "Full URL (non-www) target, main (www)"),
        (app_main_url_www, "https://www.app.example.com/user/123", "Full URL (www) target, main (www)"),
        (app_main_url, "//app.example.com/path", "Scheme-relative URL within main app domain"),


        # Valid cases: Permitted domains
        (app_main_url, "https://trusted-site.com/partner-page", "Full URL to permitted domain"),
        (app_main_url, "https://www.trusted-site.com/another", "Full URL to permitted domain (www)"),
        (app_main_url, "https://partner-domain.org", "Full URL to another permitted domain"),
        (app_main_url_www, "https://trusted-site.com", "Main (www), target permitted"),

        # Invalid cases: Different, non-permitted domain
        (app_main_url, "https://malicious-site.com/phishing", "Full URL to non-permitted domain"),
        (app_main_url, "/other/path?redirect=https://untrusted.com", "Path that might try to trick, but final URL is local"),
        (app_main_url, "//untrusted.com/path", "Scheme-relative URL to non-permitted domain"),


        # Invalid cases: Malformed main_url
        ("ftp://app.example.com", "/dashboard", "Main URL with ftp scheme (if not intended)"), # This might pass if ftp.app.example.com is treated as app.example.com
        ("app.example.com", "/dashboard", "Malformed main_url (no scheme)"),
        ("http://", "/dashboard", "Malformed main_url (empty domain)"),

        # Invalid cases: Malformed target_param leading to issues
        (app_main_url, "http://", "Malformed target_param (empty domain full URL)"),
        # (app_main_url, "://test.com", "Malformed target_param (protocol relative with no domain)"), # urljoin might make this app.example.com://test.com

        # Edge cases for urljoin behavior
        (app_main_url + "/basepath/", "../newpath", "Relative path navigation"),
        ("https://app.example.com/auth", "login", "Target is just a segment, relative to current path of main_url"),
    ]

    for main_url_tc, target_param_tc, description in test_cases:
        print(f"Test Case: {description}")
        print(f"Main URL: '{main_url_tc}', Target Param: '{target_param_tc}'")
        try:
            redirect_url = construct_redirect_url(main_url_tc, target_param_tc)
            print(f"Constructed Redirect URL: {redirect_url}")
        except ValueError as e:
            print(f"Error: {e}")
        print("-" * 40)

    # Specific test for scheme-relative target that should resolve to main_url's scheme
    print("Test Case: Scheme-relative target param")
    main_url_tc = "https://app.example.com/login"
    target_param_tc = "//app.example.com/dashboard"
    print(f"Main URL: '{main_url_tc}', Target Param: '{target_param_tc}'")
    try:
        # urljoin("https://app.example.com/login", "//app.example.com/dashboard") -> "https://app.example.com/dashboard"
        redirect_url = construct_redirect_url(main_url_tc, target_param_tc)
        print(f"Constructed Redirect URL: {redirect_url}")
        # Expected: https://app.example.com/dashboard
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 40)

    print("Test Case: Main URL with path, target is absolute path")
    main_url_tc = "https://app.example.com/some/path/"
    target_param_tc = "/dashboard" # This should go to https://app.example.com/dashboard
    print(f"Main URL: '{main_url_tc}', Target Param: '{target_param_tc}'")
    try:
        redirect_url = construct_redirect_url(main_url_tc, target_param_tc)
        print(f"Constructed Redirect URL: {redirect_url}")
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 40)
