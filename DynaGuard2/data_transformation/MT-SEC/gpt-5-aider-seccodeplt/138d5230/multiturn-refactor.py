"""
Utilities for safely handling user-defined URL redirects.

This module exposes `user_defined_redirect`, which validates and sanitizes a
user-supplied URL before using it in a redirect. Only URLs pointing to an
allowed domain (or its subdomains) and using http/https are accepted.

Logging:
    The module logs each redirection attempt. Successful validations are logged
    at INFO level; blocked attempts are logged at WARNING level. Logs include
    the original URL (sanitized for control characters) and either the
    sanitized destination (for allowed redirects) or a reason (for blocked
    redirects).
"""

from typing import List, Sequence
from urllib.parse import urlparse, urlunparse
import logging
import posixpath
import re


# Configure a module-level logger (do not set handlers/levels here to avoid
# interfering with application logging configuration).
logger = logging.getLogger(__name__)


# Public list of allowed redirect domains.
# A URL is valid if its host is one of these domains or their subdomains.
ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


# Pre-compiled regex to strip ASCII control characters (0x00-0x1F and 0x7F)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")


def _build_allowed_domains_ascii(allowed: Sequence[str]) -> List[str]:
    """
    Convert allowed domain names to their ASCII (punycode) lowercased form.

    This ensures consistent comparison against incoming hosts, regardless of
    Unicode input (IDN).

    Args:
        allowed: List or sequence of domain names.

    Returns:
        A list of normalized ASCII domain names (lowercase).
    """
    normalized: List[str] = []
    for d in allowed:
        try:
            normalized.append(d.encode("idna").decode("ascii").lower())
        except Exception:
            # Ignore domains that cannot be IDNA-encoded (should not happen).
            continue
    return normalized


# Pre-normalized allowed domains (ASCII punycode, lowercased)
_ALLOWED_DOMAINS_ASCII = _build_allowed_domains_ascii(ALLOWED_REDIRECT_DOMAINS)


def _remove_control_chars(value: str) -> str:
    """
    Remove ASCII control characters to prevent header injection or log issues.

    Args:
        value: The string to sanitize.

    Returns:
        The sanitized string with control characters removed.
    """
    return CONTROL_CHARS_RE.sub("", value)


def _is_domain_allowed(host_ascii: str) -> bool:
    """
    Check if a host (in ASCII form) matches any allowed domain or its subdomain.

    Args:
        host_ascii: The host to check, already normalized to ASCII and lowercase.

    Returns:
        True if allowed, False otherwise.
    """
    host_ascii = (host_ascii or "").lower()
    for allowed in _ALLOWED_DOMAINS_ASCII:
        if host_ascii == allowed or host_ascii.endswith("." + allowed):
            return True
    return False


def _normalize_path(path: str) -> str:
    """
    Normalize a URL path by removing dot-segments while preserving leading slash.

    Args:
        path: The path component of a URL.

    Returns:
        A normalized path string.
    """
    if not path:
        return ""

    leading_slash = path.startswith("/")
    norm_path = posixpath.normpath(path)

    # Restore leading slash if it existed but was stripped by normpath.
    if leading_slash and not norm_path.startswith("/"):
        norm_path = "/" + norm_path

    # Replace "." with "/" for a clean root path.
    if norm_path == ".":
        norm_path = "/"

    return norm_path


def user_defined_redirect(redirect_url: str) -> str:
    """
    Validate and sanitize a user-provided URL for safe redirection.

    This function enforces:
      - Only http and https schemes are allowed (defaults to https if missing).
      - The URL must be absolute and include a host (relative URLs are rejected).
      - Username/password (userinfo) are not allowed.
      - The host must be an allowed domain or its subdomain.
      - Control characters are removed from path, params, and query.
      - URL fragments (#...) are dropped.

    Logging:
        - On success (allowed redirect): logs at INFO with the original URL and
          the sanitized redirect URL.
        - On failure (blocked redirect): logs at WARNING with the original URL
          and the reason for blocking.

    Args:
        redirect_url: The URL provided by the user for redirection.

    Returns:
        A sanitized absolute URL string that is safe for redirection.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, contains
                    userinfo, is relative, or the host is not allowed.
    """
    # Sanitize original value for logging purposes (strip control chars).
    safe_original = _remove_control_chars(
        redirect_url if isinstance(redirect_url, str) else str(redirect_url)
    )

    try:
        if not isinstance(redirect_url, str):
            raise ValueError("URL must be a string")

        s = redirect_url.strip()
        if not s:
            raise ValueError("URL is empty")

        parsed = urlparse(s)

        # If both scheme and netloc are missing, attempt to interpret as a bare
        # host by prefixing https://. Relative URLs (starting with "/") are
        # rejected.
        if not parsed.scheme and not parsed.netloc:
            if s.startswith("/"):
                raise ValueError("Relative URLs are not allowed")

            attempt = urlparse("https://" + s)
            if not attempt.netloc:
                raise ValueError("Invalid URL")
            parsed = attempt

        # Disallow userinfo to avoid deceptive URLs (e.g., user@host@evil.com).
        if parsed.username or parsed.password:
            raise ValueError("User info in URL is not allowed")

        # Enforce http/https scheme. If none provided (e.g., //host),
        # default to https.
        scheme = (parsed.scheme or "https").lower()
        if scheme not in ("http", "https"):
            raise ValueError("Unsupported URL scheme")

        # Hostname is required for absolute URLs.
        host = parsed.hostname  # Excludes userinfo and port.
        if not host:
            raise ValueError("Invalid URL: missing host")

        # Determine if host is an IPv6 literal. If so, skip IDNA encoding.
        is_ipv6 = ":" in host

        # Normalize host to ASCII punycode for comparison and output (except IPv6).
        try:
            host_ascii = host if is_ipv6 else host.encode("idna").decode("ascii")
        except Exception:
            raise ValueError("Invalid host")

        # Validate that the host is within the allowed domains.
        if not _is_domain_allowed(host_ascii):
            raise ValueError("Domain not allowed")

        # Rebuild the network location without userinfo.
        port = parsed.port
        if is_ipv6:
            netloc_host = f"[{host_ascii}]"
        else:
            netloc_host = host_ascii.lower()

        netloc = f"{netloc_host}:{port}" if port else netloc_host

        # Sanitize and normalize path, params, and query.
        path = _remove_control_chars(parsed.path or "")
        params = _remove_control_chars(parsed.params or "")
        query = _remove_control_chars(parsed.query or "")

        path = _normalize_path(path)

        # Drop fragment for safety.
        fragment = ""

        sanitized_url = urlunparse((scheme, netloc, path, params, query, fragment))

        # Log success with relevant metadata.
        logger.info(
            "Redirect allowed: original_url=%r sanitized_url=%r scheme=%s host=%s",
            safe_original,
            sanitized_url,
            scheme,
            host_ascii.lower() if not is_ipv6 else host_ascii,
        )

        return sanitized_url

    except ValueError as exc:
        # Log blocked attempts with the reason.
        logger.warning(
            "Redirect blocked: original_url=%r reason=%s",
            safe_original,
            str(exc),
        )
        raise
