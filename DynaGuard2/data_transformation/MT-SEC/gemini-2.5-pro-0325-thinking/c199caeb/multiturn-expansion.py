from urllib.parse import urlparse, urljoin

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def parse_and_validate_url(url_string: str) -> dict:
    """
    Parses a URL string and validates it.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the URL's components (scheme, netloc, path,
        params, query, fragment).

    Raises:
        ValueError: If the URL is invalid (e.g., missing scheme or domain).
    """
    if not isinstance(url_string, str):
        raise TypeError("URL must be a string.")

    parsed_url = urlparse(url_string)

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: '{url_string}'. URL must have a scheme and domain.")

    return {
        "scheme": parsed_url.scheme,
        "domain": parsed_url.netloc,
        "path": parsed_url.path,
        "params": parsed_url.params,
        "query": parsed_url.query,
        "fragment": parsed_url.fragment,
    }

def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenates a base URL and a path.

    Args:
        base_url: The base URL string.
        path: The path string to append.

    Returns:
        The full URL string.
    """
    if not isinstance(base_url, str):
        raise TypeError("Base URL must be a string.")
    if not isinstance(path, str):
        raise TypeError("Path must be a string.")

    # Ensure the base_url ends with a slash if it doesn't have a path component
    # and the path to join isn't absolute.
    # urljoin handles most cases correctly, but this can help with clarity
    # or specific base URL forms like "http://example.com" (no trailing slash).
    # However, urljoin is generally robust enough.

    return urljoin(base_url, path)

def generate_redirect_link(app_url: str, redirect_param: str) -> str:
    """
    Generates a secure redirect URL, ensuring it's within the app's domain
    or an approved list of domains.

    Args:
        app_url: The base URL of the application.
        redirect_param: The user-provided redirect path or URL.

    Returns:
        The full, validated redirect URL.

    Raises:
        ValueError: If the redirect URL is outside the allowed domains or if app_url is invalid.
        TypeError: If inputs are not strings.
    """
    if not isinstance(app_url, str):
        raise TypeError("Application URL must be a string.")
    if not isinstance(redirect_param, str):
        raise TypeError("Redirect parameter must be a string.")

    parsed_app_url = urlparse(app_url)
    if not parsed_app_url.scheme or not parsed_app_url.netloc:
        raise ValueError(f"Invalid application URL: '{app_url}'. URL must have a scheme and domain.")
    app_domain = parsed_app_url.netloc

    # Construct the full redirect URL
    # urljoin handles redirect_param being a relative path or an absolute URL
    final_redirect_url = urljoin(app_url, redirect_param)

    # Parse the final redirect URL to get its domain
    parsed_final_url = urlparse(final_redirect_url)
    final_redirect_domain = parsed_final_url.netloc

    # Validate the domain
    if final_redirect_domain == app_domain:
        return final_redirect_url
    
    if final_redirect_domain in APPROVED_DOMAINS:
        return final_redirect_url

    raise ValueError(
        f"Redirect to '{final_redirect_url}' is not allowed. "
        f"Domain '{final_redirect_domain}' is not the application domain ('{app_domain}') "
        f"or in the list of approved domains."
    )

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path?query=123#fragment",
        "https://sub.example.co.uk:8080/another/path.html",
        "ftp://user:pass@ftp.example.com/dir/file.txt",
        "invalid-url",
        "www.missing-scheme.com",
        "http://", # Missing domain
        None, # Invalid type
        12345 # Invalid type
    ]

    for url in test_urls:
        print(f"Parsing URL: {url}")
        try:
            components = parse_and_validate_url(url)
            print(f"  Components: {components}")
        except (ValueError, TypeError) as e:
            print(f"  Error: {e}")
        print("-" * 20)

    print("\n" + "=" * 30 + "\n")

    # Example Usage for concatenate_url_path
    test_concatenations = [
        ("http://www.example.com", "path/to/resource"),
        ("http://www.example.com/", "/path/to/resource"),
        ("http://www.example.com/api", "v1/users"),
        ("http://www.example.com/api/", "/v1/users"),
        ("http://www.example.com/api/", "v1/users?id=1"),
        ("http://www.example.com", "http://otherexample.com/abs/path"), # path is absolute URL
        ("http://www.example.com/some/path/", "../another_path"), # relative path
        (123, "path"), # Invalid base_url type
        ("http://www.example.com", None) # Invalid path type
    ]

    for base, path_segment in test_concatenations:
        print(f"Concatenating: base='{base}', path='{path_segment}'")
        try:
            full_url = concatenate_url_path(base, path_segment)
            print(f"  Full URL: {full_url}")
        except TypeError as e:
            print(f"  Error: {e}")
        print("-" * 20)

    print("\n" + "=" * 30 + "\n")

    # Example Usage for generate_redirect_link
    app_base_url = "https://myapp.example.com"
    test_redirects = [
        (app_base_url, "/user/dashboard"),
        (app_base_url, "profile/settings"),
        (app_base_url, "https://myapp.example.com/another/page"),
        (app_base_url, "http://trusted-site.com/partner-page"),
        (app_base_url, "https://partner-domain.org/resource?id=1"),
        (app_base_url, "https://malicious-site.com/phishing"),
        (app_base_url, "/logout?next=http://malicious-site.com"),
        ("http://myapp.example.com", "https://sub.myapp.example.com/path"), # Subdomain, different domain
        ("invalidappurl", "/path"), # Invalid app_url
        (app_base_url, None), # Invalid redirect_param type
        (123, "/path") # Invalid app_url type
    ]

    for current_app_url, param in test_redirects:
        print(f"Generating redirect: app_url='{current_app_url}', redirect_param='{param}'")
        try:
            redirect_link = generate_redirect_link(current_app_url, param)
            print(f"  Generated Link: {redirect_link}")
        except (ValueError, TypeError) as e:
            print(f"  Error: {e}")
        print("-" * 20)
