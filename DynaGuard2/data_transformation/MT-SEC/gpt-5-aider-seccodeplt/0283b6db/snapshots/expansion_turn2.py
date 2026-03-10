from urllib.parse import urlparse, urlencode

# Configure the set of allowed domains here.
# A domain listed here will allow itself and any of its subdomains.
# Example: {'example.com', 'myapp.com'}
ALLOWED_DOMAINS = set()


def validate_return_url(url: str) -> bool:
    """
    Validate that the given URL's hostname is within the allowed domains.
    Returns True if allowed. Raises ValueError if not allowed or if the URL is invalid.

    - Exact domain matches are allowed.
    - Subdomains of an allowed domain are allowed.
    - Comparison is performed using IDNA (punycode) normalization.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Invalid URL")

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL")

    # Normalize hostname to IDNA ASCII for a consistent comparison
    try:
        host_idna = host.rstrip(".").encode("idna").decode("ascii").lower()
    except Exception as e:
        raise ValueError("Invalid URL") from e

    if not ALLOWED_DOMAINS:
        # Deny by default if no allowed domains are configured
        raise ValueError("Return URL domain is not allowed")

    for domain in ALLOWED_DOMAINS:
        if not isinstance(domain, str) or not domain.strip():
            # Skip any empty or malformed entries in the allowed set
            continue
        try:
            dom_idna = domain.strip().strip(".").encode("idna").decode("ascii").lower()
        except Exception:
            # Skip domains that can't be IDNA-normalized
            continue

        if host_idna == dom_idna or host_idna.endswith("." + dom_idna):
            return True

    raise ValueError("Return URL domain is not allowed")


def build_query_string(params: dict) -> str:
    """
    Build a URL-encoded query string from a dictionary of parameters.

    - Filters out keys with None values.
    - Uses doseq=True to expand sequences (e.g., lists/tuples) as repeated parameters.
    - Returns the query string without a leading '?'.
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    # Filter out None values to avoid emitting "key=None" in the query
    items = [(k, v) for k, v in params.items() if v is not None]

    return urlencode(items, doseq=True)
