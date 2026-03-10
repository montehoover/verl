from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _is_in_allowed_domains(host: str, allowed_domains) -> bool:
    host = (host or "").lower().rstrip(".")
    if not host:
        return False
    for domain in allowed_domains:
        d = str(domain).lower().lstrip(".").rstrip(".")
        if host == d or host.endswith("." + d):
            return True
    return False


def _is_within_base_domain(target_host: str, base_host: str) -> bool:
    if not target_host or not base_host:
        return False
    return target_host.lower() == base_host.lower()


def _is_relative_url(parsed_target) -> bool:
    # Relative when no scheme and no network location
    return not parsed_target.netloc and not parsed_target.scheme


def validate_redirect_target(app_base_url: str, redirect_param: str, allowed_domains=None) -> None:
    """
    Validate that redirect_param is safe relative to app_base_url or allowed external domains.

    Raises:
        ValueError: If redirect_param uses an unsupported scheme or points to a disallowed host.
    """
    if allowed_domains is None:
        allowed_domains = ALLOWED_DOMAINS

    parsed_base = urlparse.urlparse((app_base_url or "").strip())
    # Normalize backslashes in target only for parsing
    parsed_target = urlparse.urlparse(((redirect_param or "").strip()).replace("\\", "/"))

    base_host = (parsed_base.hostname or "").lower()
    target_host = (parsed_target.hostname or "").lower()

    # Only allow http/https schemes when a scheme is provided
    if parsed_target.scheme and parsed_target.scheme.lower() not in ("http", "https"):
        raise ValueError("Unsupported redirect scheme")

    # Relative targets are considered safe
    if _is_relative_url(parsed_target):
        return

    # Absolute or protocol-relative targets must be within base host or allowed domains
    if not (_is_within_base_domain(target_host, base_host) or _is_in_allowed_domains(target_host, allowed_domains)):
        raise ValueError("Disallowed external redirect target")


def construct_redirect_url(app_base_url: str, redirect_param: str) -> str:
    """
    Construct the final URL by appending next=<redirect_param> to app_base_url's query string.
    """
    parsed_base = urlparse.urlparse((app_base_url or "").strip())
    existing_params = urlparse.parse_qsl(parsed_base.query, keep_blank_values=True)
    # Remove any existing 'next' to avoid duplicates (case-insensitive)
    existing_params = [(k, v) for (k, v) in existing_params if k.lower() != "next"]
    # Append the original (un-normalized for parsing) redirect_param value
    existing_params.append(("next", (redirect_param or "").strip()))
    new_query = urlparse.urlencode(existing_params, doseq=True)
    return urlparse.urlunparse(parsed_base._replace(query=new_query))


def generate_redirect_link(app_base_url: str, redirect_param: str) -> str:
    """
    Construct a safe redirect URL for post-login navigation.

    Args:
        app_base_url: Base URL of the application where the 'next' parameter should be attached.
        redirect_param: The user-supplied path or absolute URL to navigate to after login.

    Returns:
        A string containing the final constructed redirect URL.

    Raises:
        ValueError: If redirect_param points to a domain that is neither within the base app domain
                    nor in the allowed external domains list.
    """
    base_url = (app_base_url or "").strip()
    target = (redirect_param or "").strip()

    # Validate target against base and allowed domains
    validate_redirect_target(base_url, target, ALLOWED_DOMAINS)

    # Build and return the final URL
    return construct_redirect_url(base_url, target)
