from urllib.parse import urlparse, urljoin, urlunparse
from typing import Iterable, Optional, Set


# Predefined list of allowed domains. Update this set as needed.
ALLOWED_DOMAINS: Set[str] = {"example.com", "api.example.com"}


def _extract_hostname(url: str) -> Optional[str]:
    """
    Extract the hostname from the given URL string.
    Tries parsing with and without a default scheme to handle schemeless URLs.
    Returns the hostname as a string, or None if it cannot be determined.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if host:
        return host
    # Try adding a default scheme for schemeless inputs (e.g., "example.com/path")
    parsed_with_scheme = urlparse(f"http://{url}")
    return parsed_with_scheme.hostname


def validate_url_domain(url: str, allowed_domains: Optional[Iterable[str]] = None) -> str:
    """
    Validate that the URL's domain is in the allowed list.

    Args:
        url: The URL string to validate.
        allowed_domains: Optional iterable of allowed domains. If not provided,
                         ALLOWED_DOMAINS is used.

    Returns:
        The normalized hostname (lowercase, without trailing dot) if valid.

    Raises:
        ValueError: If the input is not a non-empty string, the URL is invalid,
                    or the domain is not in the allowed list.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    hostname = _extract_hostname(url)
    if not hostname:
        raise ValueError("Invalid URL: missing or unparseable hostname")

    normalized_host = hostname.rstrip(".").lower()

    allowed = set(allowed_domains) if allowed_domains is not None else ALLOWED_DOMAINS
    allowed_normalized = {d.rstrip(".").lower() for d in allowed}

    if normalized_host not in allowed_normalized:
        raise ValueError(f"URL domain '{normalized_host}' is not allowed")

    return normalized_host


def combine_base_with_path(base: str, path: str) -> str:
    """
    Combine a base URL with a path, handling leading/trailing slashes gracefully.

    Args:
        base: The base URL.
        path: The path to append to the base.

    Returns:
        The combined URL string.

    Raises:
        ValueError: If base is not a non-empty string or path is not a string.
    """
    if not isinstance(base, str) or not base.strip():
        raise ValueError("base must be a non-empty string")
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    base = base.strip()
    path = path.strip()

    # If path is empty, return the base as-is.
    if path == "":
        return base

    # If base looks like a URL with scheme/netloc, use urljoin safely.
    parsed_base = urlparse(base)
    if parsed_base.scheme or parsed_base.netloc:
        # Ensure the base path ends with '/', and strip query/fragment for safe joining.
        base_path = parsed_base.path or ""
        if not base_path.endswith("/"):
            base_path = base_path + "/"
        base_for_join = urlunparse(parsed_base._replace(path=base_path, query="", fragment=""))
        return urljoin(base_for_join, path.lstrip("/"))

    # Handle schemeless bases (e.g., "example.com/api" or "/api")
    base_part = base.rstrip("/")
    path_part = path.lstrip("/")
    return f"{base_part}/{path_part}" if path_part else base_part
