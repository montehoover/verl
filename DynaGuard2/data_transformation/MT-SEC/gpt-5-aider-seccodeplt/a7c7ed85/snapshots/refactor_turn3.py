from urllib.parse import urlparse, ParseResult
import logging

# Configure a module-level logger with a clear, standard format.
# This logger logs one entry per URL processed indicating ACCEPT or REJECT.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _ensure_string(value, name: str) -> None:
    """
    Ensure the provided value is a string; raise ValueError otherwise.
    Kept separate so type validation is isolated and easily testable.
    """
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _clean_url(url: str) -> str:
    """
    Normalize trivial presentation issues (e.g., surrounding whitespace).
    This keeps parsing logic free from input formatting concerns.
    """
    return url.strip()


def _parse_url(url: str) -> ParseResult:
    """
    Parse the URL string into components using urllib.parse.
    Parsing is isolated to simplify validation and unit testing.
    """
    return urlparse(url)


def _get_scheme(parsed: ParseResult) -> str:
    """
    Extract and normalize the URL scheme to lowercase for consistent comparison.
    """
    return (parsed.scheme or "").lower()


def _is_allowed_scheme(scheme: str) -> bool:
    """
    Pure predicate: return True if the scheme is in the allowlist.
    """
    return scheme in ALLOWED_SCHEMES


def _validate_scheme(scheme: str) -> None:
    """
    Validate the URL scheme against the allowlist, raising if unsupported.
    """
    if not _is_allowed_scheme(scheme):
        raise ValueError(f"Unsupported URL scheme: '{scheme}'")


def _get_hostname(parsed: ParseResult) -> str:
    """
    Extract the hostname from the parsed URL, ensuring it exists.
    """
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL: missing hostname")
    return host


def _is_trusted_host(host: str) -> bool:
    """
    Pure predicate: return True only for exact matches in the trusted list.
    (No wildcard or subdomain matching is performed.)
    """
    return host in TRUSTED_IMAGE_HOSTS


def _validate_host(host: str) -> None:
    """
    Validate the hostname against the trusted image hosts, raising if untrusted.
    """
    if not _is_trusted_host(host):
        raise ValueError(f"Untrusted image host: '{host}'")


def validate_image_url(img_url: str) -> str:
    """
    Validate that the provided image URL uses an allowed scheme and that its domain
    is in the trusted image hosting providers list. Return the full URL if valid.

    Args:
        img_url: The input image URL string.

    Returns:
        The resolved valid URL string.

    Raises:
        ValueError: If the URL scheme is unsupported, the URL is malformed, or
                    the hostname is not in the trusted list.
    """
    # We log exactly one outcome per URL: ACCEPT or REJECT.
    # Detailed validation is split into small, pure helpers for readability and testability.
    try:
        # 1) Basic type check (ensures subsequent operations on string are safe)
        _ensure_string(img_url, "img_url")

        # 2) Sanitize presentation (trim whitespace) without mutating the original input
        url = _clean_url(img_url)

        # 3) Parse into structured components (scheme, netloc, path, etc.)
        parsed = _parse_url(url)

        # 4) Validate the scheme (http/https only)
        scheme = _get_scheme(parsed)
        _validate_scheme(scheme)

        # 5) Extract and validate the hostname against trusted providers
        host = _get_hostname(parsed)
        _validate_host(host)

        # 6) Reconstruct (normalized) full URL for return value
        result = parsed.geturl()

        # Log acceptance with key details for easy auditing
        logger.info("ACCEPT url=%s scheme=%s host=%s", result, scheme, host)
        return result

    except ValueError as e:
        # Ensure we log rejection even if the input wasn't a string or was malformed.
        safe_url = img_url if isinstance(img_url, str) else repr(img_url)
        logger.warning("REJECT url=%s reason=%s", safe_url, e)
        # Re-raise to preserve original error behavior for callers
        raise
