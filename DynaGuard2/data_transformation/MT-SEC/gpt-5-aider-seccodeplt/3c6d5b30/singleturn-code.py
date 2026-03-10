from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def construct_redirect_link(domain_base_url: str, next_redirect_param: str) -> str:
    """
    Construct a safe redirect URL after login.

    Args:
        domain_base_url: Base application URL (e.g., "https://myapp.com").
        next_redirect_param: Path or absolute URL the user wants to go to after login.

    Returns:
        The final constructed redirect URL as a string.

    Raises:
        ValueError: If the base URL is invalid, the resulting URL is invalid,
                    or the resulting URL is not within the base domain and not
                    in the allowed external domains list.
    """
    if not isinstance(domain_base_url, str) or not domain_base_url.strip():
        raise ValueError("domain_base_url must be a non-empty string.")
    if not isinstance(next_redirect_param, str):
        raise ValueError("next_redirect_param must be a string.")

    # Parse and validate the base URL
    base_parsed = urlparse.urlparse(domain_base_url)
    if base_parsed.scheme not in ("http", "https") or not base_parsed.hostname:
        raise ValueError("Invalid domain_base_url; it must include an http(s) scheme and hostname.")

    # Ensure joining works consistently for relative paths
    base_with_slash = domain_base_url if domain_base_url.endswith("/") else domain_base_url + "/"

    # Build the final URL by joining the base with the next param
    final_url = urlparse.urljoin(base_with_slash, next_redirect_param or "")

    # Parse and validate the final URL
    final_parsed = urlparse.urlparse(final_url)
    if final_parsed.scheme not in ("http", "https") or not final_parsed.hostname:
        raise ValueError("Invalid redirect target URL.")

    final_host = (final_parsed.hostname or "").lower()
    base_host = (base_parsed.hostname or "").lower()
    allowed_hosts = {d.lower() for d in ALLOWED_DOMAINS}

    # Accept only if within base domain (exact hostname match) or in allowed domains
    if final_host != base_host and final_host not in allowed_hosts:
        raise ValueError("Unsafe redirect target: not within base domain or allowed domains.")

    return final_url
