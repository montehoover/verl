from urllib.parse import urlsplit
from typing import Iterable

try:
    import tldextract  # type: ignore
except Exception:  # noqa: BLE001
    tldextract = None


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
    Return True if the URL's registrable domain is in the approved list, else False.
    The comparison is done on normalized domains as returned by extract_domain.
    """
    if not isinstance(url, str) or not approved_domains:
        return False

    domain = extract_domain(url)
    if not domain:
        return False

    # Handle the common pitfall where a single string is passed instead of an iterable
    if isinstance(approved_domains, str):
        candidates = [approved_domains]
    else:
        candidates = approved_domains

    normalized_set = set()
    for item in candidates:
        if not isinstance(item, str):
            continue
        item_domain = extract_domain(item)
        if item_domain:
            normalized_set.add(item_domain)

    return domain in normalized_set
