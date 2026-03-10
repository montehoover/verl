from urllib.parse import urlparse
from typing import Iterable, Optional, Set

# Predefined list of trusted domains (customize as needed)
TRUSTED_DOMAINS: Set[str] = {
    "example.com",
    "example.org",
}


def extract_url_components(url: str) -> dict:
    """
    Extract scheme, domain, and path from a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary with keys: 'scheme', 'domain', 'path'.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)

    # Handle schemeless URLs like "example.com/path"
    if not parsed.scheme and not parsed.netloc and parsed.path:
        parsed = urlparse('//' + url)

    return {
        'scheme': parsed.scheme or '',
        'domain': parsed.hostname or '',
        'path': parsed.path or '',
    }


def check_trusted_domain(url: str, trusted_domains: Optional[Iterable[str]] = None) -> str:
    """
    Check whether the URL's domain is in a predefined list of trusted domains.

    Args:
        url: The URL string to check.
        trusted_domains: Optional iterable of trusted domains. If not provided,
                         the module-level TRUSTED_DOMAINS will be used.

    Returns:
        A message indicating whether the domain is trusted or not.
    """
    components = extract_url_components(url)
    domain = (components.get('domain') or '').lower()

    if not domain:
        return "No domain found in URL."

    # Normalize trusted domains (lowercase, strip leading dots)
    trusted = {d.lower().lstrip('.') for d in (trusted_domains or TRUSTED_DOMAINS)}

    is_trusted = any(
        domain == td or domain.endswith('.' + td)
        for td in trusted
    )

    if is_trusted:
        return f"Domain '{domain}' is trusted."
    else:
        return f"Domain '{domain}' is not trusted."
