from urllib import parse as urlparse

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

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

    # Handle cases where redirect_param might be a full URL or just a path
    parsed_redirect_param = urlparse.urlparse(redirect_param)

    if parsed_redirect_param.scheme and parsed_redirect_param.netloc:
        # redirect_param is a full URL
        final_url = redirect_param
        parsed_final_url = parsed_redirect_param
    else:
        # redirect_param is a path, join it with app_url
        # Ensure redirect_param starts with a slash if it's a path component
        path_component = redirect_param
        if not path_component.startswith('/') and parsed_app_url.path.endswith('/'):
            pass # app_url ends with / and path_component doesn't start with /
        elif not path_component.startswith('/') and not parsed_app_url.path.endswith('/'):
            path_component = '/' + path_component
        elif path_component.startswith('/') and parsed_app_url.path.endswith('/'):
            path_component = path_component[1:]


        final_url = urlparse.urljoin(app_url, path_component)
        parsed_final_url = urlparse.urlparse(final_url)

    final_domain = parsed_final_url.netloc

    if final_domain == app_domain or final_domain in APPROVED_DOMAINS:
        return final_url
    else:
        raise ValueError(
            f"Redirect URL '{final_url}' is not allowed. "
            f"Must be within '{app_domain}' or one of {APPROVED_DOMAINS}."
        )
