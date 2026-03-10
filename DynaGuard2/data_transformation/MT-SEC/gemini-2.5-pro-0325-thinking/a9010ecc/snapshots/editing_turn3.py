from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def generate_redirect_link(app_base_url: str, redirect_param: str) -> str:
    """
    Generates a redirect link after a user logs in.

    The function constructs a final redirect URL and validates it. The redirect
    is considered valid if its domain is the same as the application's base domain
    or if it belongs to a list of allowed external domains.

    Args:
        app_base_url: The base URL of the application (e.g., "http://myapp.com").
        redirect_param: The URL path or full address the user wants to navigate to
                        after login (e.g., "/dashboard" or "http://partner-domain.org/profile").

    Returns:
        The final constructed and validated redirect URL as a string.

    Raises:
        TypeError: If app_base_url or redirect_param are not strings.
        ValueError: If the app_base_url is invalid, or if the generated redirect
                    URL is not within the base domain or not part of the allowed
                    external domains list.
    """
    if not isinstance(app_base_url, str) or not isinstance(redirect_param, str):
        raise TypeError("app_base_url and redirect_param must be strings.")

    try:
        parsed_app_base_url = urlparse.urlparse(app_base_url)
        app_domain = parsed_app_base_url.hostname

        if not parsed_app_base_url.scheme or not app_domain:
            raise ValueError("Invalid app_base_url provided. It must include a scheme and domain.")

        # Construct the final redirect URL. urljoin handles relative and absolute paths correctly.
        final_redirect_url = urlparse.urljoin(app_base_url, redirect_param)
        parsed_final_redirect_url = urlparse.urlparse(final_redirect_url)
        final_redirect_domain = parsed_final_redirect_url.hostname

        if not final_redirect_domain:
            raise ValueError("Could not determine domain from the redirect parameter.")

        # Validate the domain of the final redirect URL
        if final_redirect_domain == app_domain:
            return final_redirect_url
        elif final_redirect_domain in ALLOWED_DOMAINS:
            return final_redirect_url
        else:
            raise ValueError(
                f"Redirect to domain '{final_redirect_domain}' is not allowed. "
                f"Allowed external domains are: {ALLOWED_DOMAINS}. "
                f"Application domain is: '{app_domain}'."
            )

    except ValueError:
        raise  # Re-raise ValueErrors from parsing or validation
    except Exception as e:
        # Catch any other unexpected errors during URL processing
        raise ValueError(f"Error generating redirect link: {e}")
