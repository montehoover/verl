from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _construct_redirect_url(app_base_url: str, redirect_param: str) -> str:
    """Constructs the full redirect URL."""
    # Ensure app_base_url ends with a slash for proper joining if redirect_param is a path
    if not app_base_url.endswith('/'):
        app_base_url += '/'
    return urlparse.urljoin(app_base_url, redirect_param)


def _is_redirect_url_valid(redirect_url: str, app_base_url: str) -> bool:
    """Validates if the redirect URL is within the base domain or allowed external domains."""
    parsed_base_url = urlparse.urlparse(app_base_url)
    parsed_redirect_url = urlparse.urlparse(redirect_url)

    base_domain = parsed_base_url.netloc
    redirect_domain = parsed_redirect_url.netloc

    # If redirect_param was a relative path, redirect_domain might be empty.
    # In this case, it's considered to be on the same domain as app_base_url.
    if not redirect_domain:  # Indicates a relative path, so it's on the base domain
        return True

    return redirect_domain == base_domain or redirect_domain in ALLOWED_DOMAINS


def generate_redirect_link(app_base_url: str, redirect_param: str) -> str:
    """
    Generates a URL to which the user should be redirected after successfully logging in.

    It concatenates the base application URL with a 'next' parameter provided by the user
    to form the full redirect link.

    Args:
        app_base_url: A string representing the base URL of the application.
        redirect_param: A string representing the URL path or address the user
                        wants to navigate to after login.

    Returns:
        A string containing the final constructed redirect URL.

    Raises:
        ValueError: Raised if the generated redirect URL is neither within the base
                    domain nor part of the allowed external domains list.
    """
    redirect_url = _construct_redirect_url(app_base_url, redirect_param)

    if not _is_redirect_url_valid(redirect_url, app_base_url):
        # To provide a helpful error message, we need the base_domain again.
        # Alternatively, _is_redirect_url_valid could return more info or raise directly.
        # For now, let's re-parse to get base_domain for the error message,
        # or accept slight duplication in the error message logic.
        parsed_base_url = urlparse.urlparse(app_base_url)
        base_domain = parsed_base_url.netloc
        raise ValueError(
            f"Redirect URL '{redirect_url}' is not allowed. "
            f"It must be within the domain '{base_domain}' or one of {ALLOWED_DOMAINS}."
        )

    return redirect_url
