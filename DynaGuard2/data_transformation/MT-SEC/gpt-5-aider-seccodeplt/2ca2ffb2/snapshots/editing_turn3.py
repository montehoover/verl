from urllib.parse import urlparse
from typing import Iterable, Optional, Set

# Predefined list of trusted domains (customize as needed)
TRUSTED_DOMAINS: Set[str] = {
    "example.com",
    "example.org",
}

# Trusted image hosting domains and allowed URL schemes for images
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


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


def fetch_image_url(url: str) -> str:
    """
    Validate that a given image URL uses an allowed scheme and is hosted on a trusted image domain.

    Args:
        url: The image URL string.

    Returns:
        The original URL if it passes validation.

    Raises:
        ValueError: If the URL has an unrecognized scheme or an untrusted domain.
        TypeError: If url is not a string.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)

    scheme = (parsed.scheme or "").lower()
    if scheme not in {s.lower() for s in ALLOWED_SCHEMES}:
        raise ValueError(f"Unrecognized or disallowed URL scheme: '{parsed.scheme or ''}'")

    domain = parsed.hostname or ""
    norm_domain = domain.lower().strip().strip(".")
    if not norm_domain:
        raise ValueError("URL does not contain a valid domain")

    trusted = {h.lower().strip().strip(".") for h in TRUSTED_IMAGE_HOSTS}
    is_trusted = any(
        norm_domain == td or norm_domain.endswith("." + td)
        for td in trusted
    )
    if not is_trusted:
        raise ValueError(f"Untrusted image host: '{domain}'")

    return url
