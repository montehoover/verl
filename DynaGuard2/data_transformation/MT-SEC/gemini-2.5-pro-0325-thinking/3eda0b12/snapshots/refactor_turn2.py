from urllib import parse as urlparse

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _build_final_url(main_url: str, target_param: str) -> str:
    """
    Constructs the final redirect URL.
    """
    # Ensure target_param is treated as a string
    return urlparse.urljoin(main_url, str(target_param))


def _validate_redirect_url(final_url: str, main_domain: str, allowed_domains: list[str]) -> None:
    """
    Validates the final redirect URL against scheme and domain rules.

    Raises:
        ValueError: If the URL scheme is invalid or the domain is not allowed.
    """
    parsed_final_url = urlparse.urlparse(final_url)
    final_domain = parsed_final_url.netloc

    # Check if the scheme is http or https
    if parsed_final_url.scheme not in ('http', 'https'):
        raise ValueError("Invalid URL scheme: URL must be http or https.")

    # Check if the final URL is within the main application domain or a permitted domain
    if not (final_domain == main_domain or final_domain in allowed_domains):
        raise ValueError(
            f"Invalid redirect URL: '{final_url}'. "
            f"Domain '{final_domain}' is not allowed."
        )


def construct_redirect_url(main_url: str, target_param: str) -> str:
    """
    Generates a redirect URL based on a user-provided 'next' parameter.

    Args:
        main_url: The base URL of the application.
        target_param: The URL path or full URL for redirection.

    Returns:
        The final redirect URL as a string.

    Raises:
        ValueError: If the final URL is not within the application's domain
                    or an allowed external domain.
    """
    parsed_main_url = urlparse.urlparse(main_url)
    main_domain = parsed_main_url.netloc

    final_url = _build_final_url(main_url, target_param)
    
    _validate_redirect_url(final_url, main_domain, PERMITTED_DOMAINS)
    
    return final_url
