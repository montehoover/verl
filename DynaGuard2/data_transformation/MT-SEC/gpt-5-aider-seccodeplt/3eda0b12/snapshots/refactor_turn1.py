from urllib import parse as urlparse

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def construct_redirect_url(main_url: str, target_param: str) -> str:
    """
    Constructs a safe redirect URL based on a base application URL and a target parameter.
    Ensures the resulting URL remains within the application's domain or an allowed domain.

    :param main_url: Base URL of the application.
    :param target_param: URL path or full URL to redirect to after sign-in.
    :return: Final redirect URL as a string.
    :raises ValueError: If the URL is invalid or points to a disallowed domain.
    """
    if not isinstance(main_url, str) or not isinstance(target_param, str):
        raise ValueError("main_url and target_param must be strings")

    base = urlparse.urlparse(main_url)
    if base.scheme not in ("http", "https") or not base.netloc or not base.hostname:
        raise ValueError("Invalid main_url; must be an absolute HTTP(S) URL")

    # Normalize inputs
    target = target_param.strip()

    # Build the candidate redirect using RFC-compliant resolution rules
    final_url = urlparse.urljoin(main_url, target if target else "")

    parsed_final = urlparse.urlparse(final_url)

    # Only allow HTTP(S) redirects
    if parsed_final.scheme not in ("http", "https") or not parsed_final.hostname:
        raise ValueError("Disallowed redirect URL scheme or malformed URL")

    base_host = (base.hostname or "").lower()
    final_host = (parsed_final.hostname or "").lower()

    # Check if the final host is the same as the application's host
    if final_host == base_host:
        return parsed_final.geturl()

    # Allow-list check for external domains (exact match or subdomain)
    def is_permitted(host: str) -> bool:
        h = host.lower()
        for domain in PERMITTED_DOMAINS:
            d = domain.lower().lstrip(".")
            if h == d or h.endswith("." + d):
                return True
        return False

    if is_permitted(final_host):
        return parsed_final.geturl()

    raise ValueError("Redirect URL domain is not permitted")
