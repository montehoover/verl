from urllib.parse import urlparse, urlunparse

# Pre-approved external domains for safe redirections
AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def process_url_redirect(input_url: str) -> str:
    """
    Validate and sanitize a user-provided URL for safe redirection.

    Rules:
    - Only http/https schemes are allowed.
    - URL must be absolute (have a hostname).
    - No embedded userinfo (username/password) is allowed.
    - Hostname must be on the authorized domain list or a subdomain thereof.
    - Fragments are removed; query and path are preserved.
    - The returned URL is reconstructed to avoid injection issues.

    Args:
        input_url: The URL provided by the user.

    Returns:
        A validated and safe-to-use URL string for redirection.

    Raises:
        ValueError: If the URL is invalid or the hostname's domain is not authorized.
    """
    if not isinstance(input_url, str):
        raise ValueError("URL must be a string")

    # Trim surrounding whitespace and ensure no control characters
    candidate = input_url.strip()
    if not candidate or any(ord(c) < 32 for c in candidate):
        raise ValueError("Invalid URL")

    parsed = urlparse(candidate)

    # Enforce allowed schemes
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Unsupported URL scheme for redirection")

    # Must be absolute with a hostname
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("URL must be absolute and include a hostname")

    # Disallow userinfo (username:password@host)
    if parsed.username or parsed.password:
        raise ValueError("Userinfo in URL is not allowed")

    # Normalize hostname for comparison
    host = (parsed.hostname or "").lower().rstrip(".")

    # Check against authorized domains (exact match or subdomain)
    def _is_authorized(h: str) -> bool:
        for d in AUTHORIZED_REDIRECT_DOMAINS:
            dd = d.lower().rstrip(".")
            if h == dd or h.endswith("." + dd):
                return True
        return False

    if not _is_authorized(host):
        raise ValueError("The provided URL's domain is not authorized for redirection")

    # Reconstruct a safe netloc without userinfo, preserving port if non-default
    port = parsed.port
    # Add brackets for IPv6 literals when reconstructing netloc
    host_display = host
    if ":" in host_display and not host_display.startswith("["):
        host_display = f"[{host_display}]"

    # Omit default ports
    if (parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443):
        port = None

    netloc = f"{host_display}:{port}" if port else host_display

    # Rebuild URL without fragment
    safe_url = urlunparse(
        (parsed.scheme, netloc, parsed.path or "", parsed.params or "", parsed.query or "", "")
    )

    return safe_url
