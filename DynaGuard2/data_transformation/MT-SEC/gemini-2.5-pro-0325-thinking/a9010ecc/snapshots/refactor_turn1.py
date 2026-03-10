from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

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
    # Ensure app_base_url ends with a slash for proper joining if redirect_param is a path
    if not app_base_url.endswith('/'):
        app_base_url += '/'

    redirect_url = urlparse.urljoin(app_base_url, redirect_param)

    parsed_base_url = urlparse.urlparse(app_base_url)
    parsed_redirect_url = urlparse.urlparse(redirect_url)

    base_domain = parsed_base_url.netloc
    redirect_domain = parsed_redirect_url.netloc

    # If redirect_param was a relative path, redirect_domain might be empty.
    # In this case, it's considered to be on the same domain as app_base_url.
    if not redirect_domain: # Indicates a relative path, so it's on the base domain
        return redirect_url

    if redirect_domain == base_domain or redirect_domain in ALLOWED_DOMAINS:
        return redirect_url
    else:
        raise ValueError(
            f"Redirect URL '{redirect_url}' is not allowed. "
            f"It must be within the domain '{base_domain}' or one of {ALLOWED_DOMAINS}."
        )
