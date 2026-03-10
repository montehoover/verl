from urllib import parse as urlparse

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _construct_final_url(app_url: str, redirect_param: str) -> tuple[str, urlparse.ParseResult]:
    """
    Constructs the final redirect URL from the application URL and redirect parameter.
    """
    parsed_app_url = urlparse.urlparse(app_url)
    parsed_redirect_param = urlparse.urlparse(redirect_param)

    if parsed_redirect_param.scheme and parsed_redirect_param.netloc:
        # redirect_param is a full URL
        final_url = redirect_param
        parsed_final_url = parsed_redirect_param
    else:
        # redirect_param is a path, join it with app_url
        path_component = redirect_param
        if not path_component.startswith('/') and parsed_app_url.path.endswith('/'):
            pass  # app_url ends with / and path_component doesn't start with /
        elif not path_component.startswith('/') and not parsed_app_url.path.endswith('/'):
            path_component = '/' + path_component
        elif path_component.startswith('/') and parsed_app_url.path.endswith('/'):
            path_component = path_component[1:]

        final_url = urlparse.urljoin(app_url, path_component)
        parsed_final_url = urlparse.urlparse(final_url)
    return final_url, parsed_final_url


def _validate_redirect_url(final_url: str, parsed_final_url: urlparse.ParseResult, app_domain: str, allowed_domains: list[str]) -> None:
    """
    Validates if the final redirect URL is within allowed domains.
    Raises ValueError if validation fails.
    """
    final_domain = parsed_final_url.netloc
    if not (final_domain == app_domain or final_domain in allowed_domains):
        raise ValueError(
            f"Redirect URL '{final_url}' is not allowed. "
            f"Must be within '{app_domain}' or one of {allowed_domains}."
        )


def generate_redirect_link(app_url: str, redirect_param: str) -> str:
    """
    Generates a redirect URL based on a user-provided 'next' parameter.

    Args:
        app_url: The base URL of the application.
        redirect_param: The URL path or full URL for redirection.

    Returns:
        The final redirect URL as a string.

    Raises:
        ValueError: If the final URL is not within the application's domain
                    or an allowed external domain.
    """
    parsed_app_url = urlparse.urlparse(app_url)
    app_domain = parsed_app_url.netloc

    final_url, parsed_final_url = _construct_final_url(app_url, redirect_param)
    _validate_redirect_url(final_url, parsed_final_url, app_domain, APPROVED_DOMAINS)

    return final_url
