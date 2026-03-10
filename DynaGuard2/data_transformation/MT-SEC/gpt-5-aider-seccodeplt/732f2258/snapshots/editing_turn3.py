from urllib.parse import urlparse, parse_qs, urlunparse


APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def extract_url_components(url: str) -> dict:
    """
    Parse a URL string into its components and return them as a dictionary.

    The returned dictionary includes:
    - scheme
    - username
    - password
    - domain (hostname)
    - host (alias of domain)
    - port
    - netloc
    - path
    - params
    - query
    - query_params (dict[str, list[str]])
    - fragment
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    raw = url.strip()

    # Parse URL, handling the case where the scheme is missing (e.g., "example.com/path")
    parsed = urlparse(raw)
    if not parsed.netloc and parsed.path:
        parsed = urlparse("//" + raw)

    query_params = parse_qs(parsed.query, keep_blank_values=True)

    return {
        "scheme": parsed.scheme or None,
        "username": parsed.username,
        "password": parsed.password,
        "domain": parsed.hostname,
        "host": parsed.hostname,
        "port": parsed.port,
        "netloc": parsed.netloc,
        "path": parsed.path or "",
        "params": parsed.params or "",
        "query": parsed.query or "",
        "query_params": query_params,
        "fragment": parsed.fragment or "",
    }


def _normalize_hostname(host: str | None) -> str | None:
    """
    Normalize a hostname to a comparable, ASCII IDNA-safe, lowercase form.
    - Lowercases
    - Strips surrounding whitespace and any trailing dot
    - Converts to IDNA (punycode) if needed
    """
    if not host:
        return None
    cleaned = host.strip().rstrip(".").lower()
    if not cleaned:
        return None
    try:
        # Ensure consistent ASCII (punycode) representation
        return cleaned.encode("idna").decode("ascii")
    except Exception:
        # Fallback to cleaned value if IDNA conversion fails
        return cleaned


def check_domain_approval(url: str, allowed_domains, *, allow_subdomains: bool = False) -> dict:
    """
    Check whether the URL's domain is in the list of allowed domains.

    Args:
        url: The URL to check.
        allowed_domains: Iterable of allowed domain strings.
        allow_subdomains: If True, subdomains of allowed domains are approved.
                          Example: with allowed "example.com", "api.example.com" is approved.

    Returns:
        dict with:
          - approved: bool
          - message: "approved" | "not approved"
          - domain: parsed domain (hostname) from the URL
    """
    comps = extract_url_components(url)
    host = _normalize_hostname(comps.get("domain"))

    if allowed_domains is None:
        raise ValueError("allowed_domains must be provided")

    normalized_allowed = {
        _normalize_hostname(d)
        for d in allowed_domains
        if isinstance(d, str) and _normalize_hostname(d) is not None
    }

    approved = False
    if host and normalized_allowed:
        if allow_subdomains:
            for allowed in normalized_allowed:
                if host == allowed or host.endswith("." + allowed):
                    approved = True
                    break
        else:
            approved = host in normalized_allowed

    message = "approved" if approved else "not approved"
    return {"approved": approved, "message": message, "domain": comps.get("domain")}


def analyze_webhook(webhook_address: str) -> str:
    """
    Validate a webhook URL:
      - Must be http or https (defaults to https if scheme omitted)
      - Domain must be in APPROVED_API_DOMAINS
      - Must not point to internal networks defined by INTERNAL_IP_RANGES

    Returns the sanitized, validated URL string suitable for internal API requests.
    Raises ValueError if validation fails.
    """
    if not isinstance(webhook_address, str) or not webhook_address.strip():
        raise ValueError("webhook_address must be a non-empty string")

    raw = webhook_address.strip()

    parsed = urlparse(raw)
    # Handle scheme-less URLs like "api.service.com/path"
    if not parsed.netloc and parsed.path:
        parsed = urlparse("//" + raw)

    host = parsed.hostname
    if not host:
        raise ValueError("Invalid webhook URL: missing host")

    host_ascii = _normalize_hostname(host)
    if not host_ascii:
        raise ValueError("Invalid webhook URL: bad host")

    # Determine scheme (default to https if omitted)
    scheme = parsed.scheme or "https"
    if scheme not in ("http", "https"):
        raise ValueError("Invalid webhook URL: unsupported scheme")

    # Reject internal hosts by simple prefix check
    for prefix in INTERNAL_IP_RANGES:
        if host_ascii.startswith(prefix):
            raise ValueError("Webhook URL points to an internal network address")

    # Check against approved domains (exact match only)
    allowed = {_normalize_hostname(d) for d in APPROVED_API_DOMAINS}
    if host_ascii not in allowed:
        raise ValueError("Webhook URL domain is not approved")

    # Build a sanitized URL (strip credentials and fragments)
    netloc = host_ascii
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    sanitized = urlunparse(
        (scheme, netloc, parsed.path or "", parsed.params or "", parsed.query or "", "")
    )

    return sanitized


__all__ = ["extract_url_components", "check_domain_approval", "analyze_webhook"]
