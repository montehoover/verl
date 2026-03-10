from urllib.parse import urlsplit, urlparse
from typing import Iterable

try:
    import tldextract  # type: ignore
except Exception:  # noqa: BLE001
    tldextract = None

# Pre-approved external domains for safe redirections
VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def _ensure_url_has_netloc(url: str) -> str:
    # If the URL lacks a scheme, urlsplit will put the host in the path.
    # Prepend '//' so it gets parsed into netloc/hostname.
    if "://" in url:
        return url
    return f"//{url}"


def _idna_decode(host: str) -> str:
    # Convert punycode to unicode if applicable; otherwise return lowercased host.
    try:
        return host.encode("ascii", "strict").decode("idna").lower()
    except Exception:
        return host.lower()


def extract_domain(url: str) -> str:
    """
    Extract the registrable domain from a URL.
    - If tldextract is available, uses it to return the registered domain (e.g., example.co.uk).
    - Otherwise, returns the hostname from the URL (lowercased and IDNA-decoded).
    """
    if not isinstance(url, str):
        return ""

    url = url.strip()
    if not url:
        return ""

    # Prefer tldextract when available
    if tldextract is not None:
        try:
            ext = tldextract.extract(url)
            # For hosts like 'localhost' or IPs, ext.domain and ext.suffix may be empty
            if ext.domain and ext.suffix:
                return f"{ext.domain}.{ext.suffix}".lower()
            elif ext.domain:
                return ext.domain.lower()
            else:
                # Fallback to hostname parsing below
                pass
        except Exception:
            # Fallback to hostname parsing below
            pass

    # Fallback: use urllib.parse to get the hostname
    try:
        parts = urlsplit(_ensure_url_has_netloc(url))
        host = parts.hostname or ""
    except Exception:
        host = ""

    if not host:
        return ""

    return _idna_decode(host)


def is_approved_domain(url: str, approved_domains: Iterable[str]) -> bool:
    """
    Return True if the URL's registrable domain (or host when PSL isn't available)
    is in the approved list (including subdomains), else False.
    The comparison is done on normalized domains/hosts.
    """
    if not isinstance(url, str) or not approved_domains:
        return False

    # Handle the common pitfall where a single string is passed instead of an iterable
    if isinstance(approved_domains, str):
        candidates = [approved_domains]
    else:
        candidates = approved_domains

    # If tldextract is available, compare registrable domains directly.
    if tldextract is not None:
        domain = extract_domain(url)
        if not domain:
            return False

        normalized_set = set()
        for item in candidates:
            if not isinstance(item, str):
                continue
            item_domain = extract_domain(item)
            if item_domain:
                normalized_set.add(item_domain)

        return domain in normalized_set

    # Fallback when tldextract isn't available:
    # Compare normalized host against approved domains allowing subdomains.
    try:
        parts = urlsplit(_ensure_url_has_netloc(url))
        host = parts.hostname or ""
    except Exception:
        host = ""
    host = _idna_decode(host) if host else ""
    if not host:
        return False

    normalized_approved = set()
    for item in candidates:
        if not isinstance(item, str):
            continue
        try:
            iparts = urlsplit(_ensure_url_has_netloc(item))
            candidate_host = iparts.hostname or item
        except Exception:
            candidate_host = item
        candidate_host = _idna_decode(candidate_host)
        if candidate_host:
            normalized_approved.add(candidate_host)

    for approved in normalized_approved:
        if host == approved or host.endswith("." + approved):
            return True
    return False


def secure_user_redirect(url_input: str) -> str:
    """
    Validate a user-provided URL for safe redirection.
    - Relative URLs (no scheme and no netloc) are allowed and returned as-is.
    - Absolute URLs must use http/https and belong to an approved domain.
    Raises ValueError if the URL is invalid or unapproved.
    """
    if not isinstance(url_input, str):
        raise ValueError("Invalid URL")

    url_input = url_input.strip()
    if not url_input:
        raise ValueError("Invalid URL")

    parsed = urlparse(url_input)

    # Allow relative URLs (treated as internal redirects)
    if not parsed.scheme and not parsed.netloc:
        return url_input

    # For absolute URLs, only http/https are allowed and netloc must exist
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise ValueError("Disallowed or malformed URL scheme")

    if not is_approved_domain(url_input, VALID_REDIRECT_DOMAINS):
        raise ValueError("Unapproved redirect domain")

    return url_input
