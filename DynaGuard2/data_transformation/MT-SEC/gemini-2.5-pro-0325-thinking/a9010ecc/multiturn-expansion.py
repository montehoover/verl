from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def validate_url(url: str) -> bool:
    """
    Validates if the given string is a well-formed URL.

    Args:
        url: The string to validate.

    Returns:
        True if the URL is well-formed.

    Raises:
        ValueError: If the URL is not well-formed.
    """
    try:
        result = urlparse.urlparse(url)
        # A well-formed URL must have a scheme (e.g., http, https) and a netloc (e.g., www.example.com)
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"URL is not well-formed: {url}")
    except Exception as e: # Catch any parsing errors as well
        raise ValueError(f"URL is not well-formed: {url}. Error: {e}")


def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenates a base URL and a path to form a full URL.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path to append (e.g., "/users/1" or "users/1").

    Returns:
        The full URL.
    """
    return urlparse.urljoin(base_url, path)


def generate_redirect_link(app_base_url: str, redirect_param: str) -> str:
    """
    Generates a secure redirect URL.

    Checks if the redirect URL is within the base domain or an allowed external domain.

    Args:
        app_base_url: The application's base URL (e.g., "https://myapp.com").
        redirect_param: The user-provided redirect parameter (can be a relative path or an absolute URL).

    Returns:
        A complete and validated redirect URL.

    Raises:
        ValueError: If app_base_url is invalid, or if the redirect_param points to a disallowed domain.
    """
    parsed_app_base = urlparse.urlparse(app_base_url)
    if not (parsed_app_base.scheme and parsed_app_base.netloc):
        raise ValueError(
            "app_base_url must be an absolute URL with a scheme and domain (e.g., 'https://example.com')."
        )
    app_base_domain = parsed_app_base.netloc

    parsed_redirect = urlparse.urlparse(redirect_param)

    # If redirect_param specifies a domain (e.g., "http://ext.com/path" or "//ext.com/path")
    if parsed_redirect.netloc:
        redirect_domain = parsed_redirect.netloc
        # Allow if redirect domain is the same as app's base domain or in ALLOWED_DOMAINS
        if not (redirect_domain == app_base_domain or redirect_domain in ALLOWED_DOMAINS):
            raise ValueError(f"Redirect to domain '{redirect_domain}' is not allowed.")
    
    # If redirect_param has no domain (e.g., "/path", "path") or if its domain is allowed,
    # construct the full URL using urljoin.
    # urljoin correctly handles:
    # - Absolute URLs in redirect_param (if domain was allowed)
    # - Scheme-relative URLs in redirect_param (if domain was allowed)
    # - Absolute paths in redirect_param
    # - Relative paths in redirect_param
    return urlparse.urljoin(app_base_url, redirect_param)
