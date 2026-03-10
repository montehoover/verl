from urllib.parse import urlparse, parse_qs


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


__all__ = ["extract_url_components", "check_domain_approval"]
