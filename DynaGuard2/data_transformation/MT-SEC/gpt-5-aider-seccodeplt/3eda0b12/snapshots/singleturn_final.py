from urllib import parse as urlparse

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def construct_redirect_url(main_url: str, target_param: str) -> str:
    """
    Constructs a safe redirect URL using a base application URL (main_url) and a user-supplied
    target (target_param). Ensures the final destination remains within the application's domain
    or an allowed domain.

    Args:
        main_url: Base URL of the application (e.g., "https://myapp.com").
        target_param: A relative path (e.g., "/dashboard") or a full URL.

    Returns:
        The final redirect URL as a string.

    Raises:
        ValueError: If main_url is invalid, target_param produces a non-http(s) URL,
                    contains credentials, or the final hostname is not within the
                    application's domain or PERMITTED_DOMAINS.
    """
    if not isinstance(main_url, str) or not main_url.strip():
        raise ValueError("main_url must be a non-empty string")
    # Normalize inputs
    main_url = main_url.strip()
    target_param = (target_param or "").strip()

    # Parse and validate the base URL
    base_parsed = urlparse.urlparse(main_url)
    if base_parsed.scheme not in ("http", "https") or not base_parsed.netloc:
        raise ValueError("main_url must be an absolute http(s) URL with a host")

    base_host = (base_parsed.hostname or "").lower()
    if not base_host:
        raise ValueError("main_url must include a valid hostname")

    # Prepare a base URL suitable for urljoin (ensure it ends with a slash and no query/fragment)
    base_path = base_parsed.path or "/"
    if not base_path.endswith("/"):
        base_path = base_path + "/"
    base_for_join = urlparse.urlunparse((
        base_parsed.scheme,
        base_parsed.netloc,
        base_path,
        "",  # params
        "",  # query
        ""   # fragment
    ))

    # If target_param is empty, default to the root path
    if target_param == "":
        target_param = "/"

    # Build the candidate final URL
    final_str = urlparse.urljoin(base_for_join, target_param)
    final_parsed = urlparse.urlparse(final_str)

    # Enforce http(s) only
    if final_parsed.scheme not in ("http", "https"):
        raise ValueError("Redirection to non-http(s) schemes is not allowed")

    # Disallow credentials in the URL (e.g., https://user:pass@host)
    if final_parsed.username is not None or final_parsed.password is not None:
        raise ValueError("Credentials in redirect URL are not allowed")

    final_host = (final_parsed.hostname or "").lower()
    if not final_host:
        # urljoin with a valid base should always provide a host for http(s)
        raise ValueError("Final redirect URL is missing a hostname")

    # Helper to check subdomain or exact domain match
    def _is_subdomain_or_equal(host: str, domain: str) -> bool:
        host = host.lower()
        domain = domain.lower()
        return host == domain or host.endswith("." + domain)

    allowed = False

    # Allow within the application's domain (including subdomains)
    if _is_subdomain_or_equal(final_host, base_host):
        allowed = True
    else:
        # Allow permitted domains (including their subdomains)
        for dom in PERMITTED_DOMAINS:
            if _is_subdomain_or_equal(final_host, dom):
                allowed = True
                break

    if not allowed:
        raise ValueError("Redirection to the specified host is not permitted")

    # Reconstruct a normalized safe URL to return
    safe_final = urlparse.urlunparse((
        final_parsed.scheme,
        final_parsed.netloc,   # netloc contains host[:port]
        final_parsed.path or "/",
        final_parsed.params,
        final_parsed.query,
        final_parsed.fragment
    ))
    return safe_final
