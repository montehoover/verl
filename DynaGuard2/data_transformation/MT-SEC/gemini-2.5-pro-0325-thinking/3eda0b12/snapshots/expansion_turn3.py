from urllib.parse import urlparse, urljoin

# Global variable for permitted domains
PERMITTED_DOMAINS = { # Updated as per new requirements
    "trusted-site.com",
    "partner-domain.org",
}

# Helper function to extract and normalize domain
def get_domain_from_url(url_str: str) -> str | None:
    """
    Parses a URL string and returns its normalized domain (netloc, lowercased, port removed).
    Returns None if parsing fails or if the URL has no network location (e.g. mailto:, file:).
    """
    if not url_str: # Early exit for empty string
        return None
    try:
        # urlparse can handle various schemes, ensure it has a netloc for "domain"
        parsed_url = urlparse(url_str)
        if not parsed_url.netloc: # No network location part
            return None
            
        domain = parsed_url.netloc
        # Remove port if present
        if ":" in domain:
            domain_part, port_part = domain.split(":", 1)
            # Basic validation for port part to avoid issues if malformed like "domain:"
            # If port_part is empty (e.g. "domain.com:"), domain_part is "domain.com"
            domain = domain_part
        
        return domain.lower() # Normalize to lowercase
    except ValueError: # Handles issues like invalid IPv6 addresses in netloc, etc.
        return None 
    except Exception: # Catch any other unexpected parsing issue
        return None

def is_domain_permitted(url: str) -> bool:
    """
    Checks if the domain of the given URL is in the list of permitted domains.

    Args:
        url: The URL string to check.

    Returns:
        True if the domain is permitted, False otherwise.
    """
    try:
        validate_url(url) # Ensure URL is structurally valid first
        domain = get_domain_from_url(url)
        
        if not domain:
            # This can happen if URL is valid but has no domain (e.g. mailto:test@example.com)
            # or if get_domain_from_url returns None for some other reason.
            return False

        if domain in PERMITTED_DOMAINS:
            return True
        # Handle www. subdomain for permitted domains
        if domain.startswith("www.") and domain[4:] in PERMITTED_DOMAINS:
            return True
            
        return False
    except ValueError: # From validate_url if URL is malformed
        return False

def construct_redirect_url(main_url_str: str, target_param_str: str) -> str:
    """
    Constructs a secure redirect URL by joining main_url_str with target_param_str.
    The target_param_str can be a relative path or an absolute URL.
    The final resolved redirect URL must be to the same domain as main_url_str
    or to a domain listed in PERMITTED_DOMAINS.

    Args:
        main_url_str: The base URL of the current application context.
        target_param_str: The target path or URL for redirection.

    Returns:
        A complete and validated redirect URL string.

    Raises:
        ValueError: If main_url_str is empty, if the resulting URL is malformed,
                    or if the redirection target domain is not permitted.
    """
    if not main_url_str: # Basic check for main_url_str
        raise ValueError("Main URL cannot be empty.")
    
    # 1. Construct the potential redirect URL using urljoin
    # urljoin handles target_param_str being relative or absolute correctly.
    # e.g., urljoin("http://a.com/b/c", "/d") -> "http://a.com/d"
    # e.g., urljoin("http://a.com/b/c", "http://x.com") -> "http://x.com"
    potential_redirect_url = urljoin(main_url_str, target_param_str)

    # 2. Validate the structure of the potential redirect URL
    # validate_url will raise ValueError if it's malformed (e.g. missing scheme, netloc for http/https)
    validate_url(potential_redirect_url)

    # 3. Get domains for comparison
    main_domain = get_domain_from_url(main_url_str)
    if not main_domain:
        # This implies main_url_str, despite being non-empty, doesn't have a parseable domain.
        # This could be due to it being a relative path itself, or an unsupported scheme.
        raise ValueError(f"Could not parse domain from main_url: {main_url_str}")

    redirect_domain = get_domain_from_url(potential_redirect_url)
    if not redirect_domain:
        # This case implies potential_redirect_url (which passed validate_url)
        # resulted in a URL type for which a domain is not applicable (e.g. mailto, file)
        # or get_domain_from_url failed for an edge case.
        # validate_url checks for scheme AND netloc, so this should be rare for http/https.
        raise ValueError(f"Could not parse domain from potential_redirect_url: {potential_redirect_url}")

    # 4. Check if the redirect domain is permitted
    is_permitted = False
    if redirect_domain == main_domain:
        is_permitted = True
    else:
        # Check against PERMITTED_DOMAINS (handling 'www.')
        if redirect_domain in PERMITTED_DOMAINS:
            is_permitted = True
        elif redirect_domain.startswith("www.") and redirect_domain[4:] in PERMITTED_DOMAINS:
            is_permitted = True
    
    if not is_permitted:
        raise ValueError(
            f"Redirect to domain '{redirect_domain}' (from URL '{potential_redirect_url}') is not permitted. "
            f"Allowed domains are '{main_domain}' or one of {list(PERMITTED_DOMAINS)}."
        )

    # 5. If all checks pass, return the constructed and validated URL
    return potential_redirect_url

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
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"Invalid URL: {url}. Missing scheme or network location.")
    except Exception as e: # Catch any parsing errors from urlparse itself, though less common for basic structure
        raise ValueError(f"Invalid URL: {url}. Parsing error: {e}")

if __name__ == '__main__':
    # Example Usage
    valid_urls = [
        "http://www.example.com",
        "https://example.com/path?query=value#fragment",
        "ftp://user:password@host:port/path",
    ]
    invalid_urls = [
        "www.example.com",
        "example.com",
        "http//example.com",
        "just_a_string",
        "",
        None, # type: ignore
        "http://",
        "http:///path",
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
        except TypeError as e: # Handles None case specifically if type hints are enforced at runtime
             print(f"Correctly identified '{url_str}' as invalid (TypeError): {e}")

    print("\nTesting domain permissions (with updated PERMITTED_DOMAINS):")
    # Updated test cases for is_domain_permitted based on new PERMITTED_DOMAINS
    permitted_domain_urls_for_is_domain_permitted = [ # Renamed to avoid clash
        "http://trusted-site.com/page",
        "https://www.trusted-site.com/another/path", # www version
        "http://partner-domain.org",
        "ftp://trusted-site.com/resource", # FTP scheme with permitted domain
        "http://trusted-site.com:8080/path" # Domain check should ignore port
    ]
    non_permitted_domain_urls_for_is_domain_permitted = [ # Renamed
        "http://untrusted.com",
        "https://sub.untrusted.org/path", # Different subdomain not in PERMITTED_DOMAINS
        "http://example.com", # No longer in PERMITTED_DOMAINS
        "www.partner-domain.org/path", # Invalid URL structure, caught by validate_url
        "http://other-trusted-site.com", # Not in PERMITTED_DOMAINS
        "mailto:test@trusted-site.com", # Valid URL, but no domain for is_domain_permitted context
    ]

    print("\nTesting permitted domain URLs (is_domain_permitted):")
    for url_str in permitted_domain_urls_for_is_domain_permitted:
        try:
            if is_domain_permitted(url_str):
                print(f"PASSED: Domain for '{url_str}' is permitted.")
            else:
                # This path might be hit if validate_url passes but get_domain_from_url returns None (e.g. mailto)
                # or if logic in is_domain_permitted has an issue.
                print(f"FAILED: Domain for '{url_str}' is NOT permitted (but was expected to be).")
        except ValueError as e: # Should not be hit if is_domain_permitted handles ValueError from validate_url
            print(f"ERROR: Validation error for '{url_str}': {e} (expected permitted).")


    print("\nTesting non-permitted domain URLs (is_domain_permitted):")
    for url_str in non_permitted_domain_urls_for_is_domain_permitted:
        # is_domain_permitted is expected to return False for these, either due to domain mismatch or invalid URL
        if not is_domain_permitted(url_str):
            print(f"PASSED: Domain for '{url_str}' is correctly not permitted.")
        else:
            print(f"FAILED: Domain for '{url_str}' IS permitted (but was expected NOT to be).")


    print("\nTesting construct_redirect_url:")
    base_app_url = "https://myapp.com/user/settings"
    
    test_cases_construct_redirect = [
        # Valid cases: same domain
        (base_app_url, "/profile", f"https://myapp.com/profile"),
        (base_app_url, "edit?section=email", f"https://myapp.com/user/edit?section=email"),
        (base_app_url, "https://myapp.com/dashboard", "https://myapp.com/dashboard"),
        (base_app_url, "/valid path with spaces", "https://myapp.com/valid%20path%20with%20spaces"),

        # Valid cases: permitted external domains
        (base_app_url, "https://trusted-site.com/partner-page", "https://trusted-site.com/partner-page"),
        (base_app_url, "http://www.partner-domain.org/info", "http://www.partner-domain.org/info"), # www version of a permitted domain
        (base_app_url, "http://partner-domain.org/info", "http://partner-domain.org/info"),


        # Invalid cases: non-permitted external domains
        (base_app_url, "https://evil-site.com/phishing", "ValueError"),
        (base_app_url, "http://another-app.com/feature", "ValueError"),
        # Check www subdomain of main_app_url if not explicitly in PERMITTED_DOMAINS
        # main_domain is 'myapp.com', redirect_domain 'www.myapp.com'. Not equal.
        # 'www.myapp.com' (or 'myapp.com') is not in PERMITTED_DOMAINS. So, ValueError.
        (base_app_url, "https://www.myapp.com/dashboard", "ValueError"),


        # Invalid cases: malformed target_param leading to malformed potential_redirect_url
        (base_app_url, "http//malformed-url", "ValueError"), # validate_url should catch this
        
        # Invalid case: main_url issues
        ("ftp://user@malformed_main/", "/path", "ValueError"), # main_url domain parsing might fail
        ("", "/path", "ValueError"), # Empty main_url
        ("not_a_url", "/path", "ValueError"), # main_url not valid for domain extraction

        # Invalid case: target leads to URL type not having a domain (e.g. mailto)
        # urljoin("https://myapp.com", "mailto:test@example.com") -> "mailto:test@example.com"
        # validate_url("mailto:test@example.com") -> True (it's a valid URL structure)
        # get_domain_from_url("mailto:test@example.com") -> None
        # construct_redirect_url will raise ValueError because redirect_domain is None.
        (base_app_url, "mailto:test@example.com", "ValueError"),
    ]

    for main_url, target, expected in test_cases_construct_redirect:
        print(f"\nTesting: main_url='{main_url}', target='{target}'")
        try:
            redirect_url = construct_redirect_url(main_url, target)
            if expected == "ValueError":
                print(f"  FAILED: Expected ValueError, but got '{redirect_url}'")
            elif redirect_url == expected:
                print(f"  PASSED: Result is '{redirect_url}'")
            else:
                print(f"  FAILED: Expected '{expected}', but got '{redirect_url}'")
        except ValueError as e:
            if expected == "ValueError":
                print(f"  PASSED: Correctly got ValueError: {str(e).splitlines()[0]}") # Print first line of error
            else:
                print(f"  FAILED: Expected '{expected}', but got ValueError: {e}")
        except Exception as e:
            print(f"  ERROR: Unexpected exception: {type(e).__name__} - {e}")

    # Test case for main_url without scheme, urljoin behavior
    print(f"\nTesting: main_url='myapp.com/user/settings' (no scheme), target='/profile'")
    try:
        # urljoin('myapp.com/user/settings', '/profile') -> 'myapp.com/profile' (relative path)
        # validate_url will fail this as it needs a scheme for http/https type validation.
        redirect_url = construct_redirect_url('myapp.com/user/settings', '/profile')
        print(f"  FAILED: Expected ValueError for main_url without scheme, but got '{redirect_url}'")
    except ValueError as e:
        print(f"  PASSED: Correctly got ValueError for main_url without scheme: {str(e).splitlines()[0]}")
    except Exception as e:
        print(f"  ERROR: Unexpected exception: {type(e).__name__} - {e}")
