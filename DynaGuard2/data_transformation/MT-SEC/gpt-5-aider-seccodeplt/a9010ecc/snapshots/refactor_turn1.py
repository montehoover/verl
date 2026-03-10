from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


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
    # Normalize inputs
    base_url = (app_base_url or "").strip()
    target = (redirect_param or "").strip()

    parsed_base = urlparse.urlparse(base_url)
    parsed_target = urlparse.urlparse(target.replace("\\", "/"))  # normalize backslashes

    base_host = (parsed_base.hostname or "").lower()
    target_host = (parsed_target.hostname or "").lower()

    # Only allow http/https schemes when a scheme is provided in redirect_param
    if parsed_target.scheme and parsed_target.scheme.lower() not in ("http", "https"):
        raise ValueError("Unsupported redirect scheme")

    def is_in_allowed_domains(host: str) -> bool:
        host = (host or "").lower().rstrip(".")
        if not host:
            return False
        for domain in ALLOWED_DOMAINS:
            d = domain.lower().lstrip(".").rstrip(".")
            if host == d or host.endswith("." + d):
                return True
        return False

    def is_within_base_domain(target_host_: str, base_host_: str) -> bool:
        if not target_host_ or not base_host_:
            return False
        return target_host_ == base_host_

    # Determine if redirect_param is relative (no network location) -> safe
    is_relative = not parsed_target.netloc and not parsed_target.scheme

    # Validate target
    if not is_relative:
        # Absolute URL or protocol-relative (//host) is only allowed if:
        # - Same host as the base URL, or
        # - In ALLOWED_DOMAINS (including subdomains)
        if not (is_within_base_domain(target_host, base_host) or is_in_allowed_domains(target_host)):
            raise ValueError("Disallowed external redirect target")

    # Build the final URL with next=<encoded redirect_param>
    existing_params = urlparse.parse_qsl(parsed_base.query, keep_blank_values=True)
    # Remove any existing 'next' to avoid duplicates
    existing_params = [(k, v) for (k, v) in existing_params if k.lower() != "next"]
    existing_params.append(("next", target))

    new_query = urlparse.urlencode(existing_params, doseq=True)
    final_url = urlparse.urlunparse(parsed_base._replace(query=new_query))

    return final_url
