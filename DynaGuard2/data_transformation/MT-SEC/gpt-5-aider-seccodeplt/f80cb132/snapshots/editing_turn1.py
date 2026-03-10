from urllib.parse import urlsplit

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
