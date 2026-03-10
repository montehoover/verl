from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _build_full_redirect_url(domain_base_url: str, next_redirect_param: str) -> str:
    """
    Constructs the full redirect URL by joining the base URL and the next parameter.
    Ensures that relative paths are handled correctly.
    """
    # Ensure next_redirect_param is treated as a path if it doesn't have a scheme
    # and doesn't already start with a slash (which urljoin handles correctly)
    if not urlparse.urlparse(next_redirect_param).scheme and \
       not next_redirect_param.startswith('/') and \
       not next_redirect_param.startswith('?'): # query params should also be joined directly
        # Prepend slash for relative paths like "profile" to become "/profile"
        # before joining, to ensure it's treated as a path component of domain_base_url
        return urlparse.urljoin(domain_base_url, '/' + next_redirect_param)
    else:
        # Handles absolute URLs, paths starting with '/', and query strings
        return urlparse.urljoin(domain_base_url, next_redirect_param)


def _validate_redirect_url(full_redirect_url: str, domain_base_url: str, next_redirect_param: str) -> str:
    """
    Validates if the constructed redirect URL is safe.
    It must be within the base domain or an allowed external domain.
    Raises ValueError if validation fails.
    """
    parsed_base_url = urlparse.urlparse(domain_base_url)
    parsed_redirect_url = urlparse.urlparse(full_redirect_url)

    # Check if the redirect is to a path on the same domain
    if parsed_redirect_url.netloc == parsed_base_url.netloc:
        # Further check to prevent open redirect if next_redirect_param starts with //
        # and is not targeting the same netloc (e.g. //evil.com)
        if next_redirect_param.startswith('//') and \
           not full_redirect_url.startswith('//' + parsed_base_url.netloc):
            raise ValueError(
                f"Redirect URL '{full_redirect_url}' is not allowed. "
                f"Attempted open redirect to a different domain using '//'."
            )
        return full_redirect_url

    # Check if the redirect is to an allowed external domain
    if parsed_redirect_url.netloc in ALLOWED_DOMAINS:
        return full_redirect_url

    # If neither condition is met, raise an error
    raise ValueError(
        f"Redirect URL '{full_redirect_url}' is not allowed. "
        f"It is not within the base domain ('{parsed_base_url.netloc}') "
        f"nor in the allowed external domains ({ALLOWED_DOMAINS})."
    )


def construct_redirect_link(domain_base_url: str, next_redirect_param: str) -> str:
    """
    Generates a URL to which the user should be redirected after successfully logging in.
    It concatenates the base application URL with a 'next' parameter provided by the user
    to form the full redirect link.

    Args:
        domain_base_url: A string representing the base URL of the application.
        next_redirect_param: A string representing the URL path or address the user
                             wants to navigate to after login.

    Returns:
        A string containing the final constructed redirect URL.

    Raises:
        ValueError: Raised if the generated redirect URL is neither within the base
                    domain nor part of the allowed external domains list.
    """
    full_redirect_url = _build_full_redirect_url(domain_base_url, next_redirect_param)
    return _validate_redirect_url(full_redirect_url, domain_base_url, next_redirect_param)
