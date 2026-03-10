from urllib.parse import urlparse, urlunparse
from typing import Sequence
import logging

logger = logging.getLogger(__name__)

VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def _idna_label(value: str) -> str:
    """
    Normalize a hostname label to its ASCII IDNA form.
    """
    try:
        return value.encode("idna").decode("ascii")
    except Exception:
        return value


def is_host_in_allowed_domains(hostname: str, allowed_domains: Sequence[str]) -> bool:
    """
    Pure function that checks whether the given hostname is within the allowed domains.
    It performs normalization (lowercasing, stripping trailing dot, and IDNA encoding)
    before comparison. Subdomains are allowed.

    Returns True if allowed, False otherwise.
    """
    if not hostname:
        return False

    host = _idna_label(hostname.lower().rstrip("."))

    for domain in allowed_domains:
        dom = _idna_label(str(domain).lower().rstrip("."))
        if host == dom or host.endswith("." + dom):
            return True

    return False


def secure_user_redirect(url_input: str) -> str:
    """
    Validate and normalize a user-provided URL for safe redirection.

    Logs each attempt with:
    - original URL input
    - validation result (success/failure)
    - final redirection URL on success

    Rules:
    - Allow http/https schemes only.
    - If absolute URL: hostname must be within VALID_REDIRECT_DOMAINS (or a subdomain).
    - Disallow embedded credentials (user:pass@).
    - Allow relative URLs (treated as internal); normalize to start with "/".
    - Reject control characters and header injection vectors (CR/LF, NUL).
    """
    original_input = url_input
    try:
        if not isinstance(url_input, str):
            raise ValueError("URL must be a string")

        s = url_input.strip()
        if not s:
            raise ValueError("URL cannot be empty")

        # Basic injection safeguards
        if any(ch in s for ch in ("\r", "\n", "\x00")):
            raise ValueError("URL contains invalid control characters")

        # Reject backslashes to avoid ambiguous interpretation across agents
        if "\\" in s:
            raise ValueError("URL contains invalid backslash characters")

        parsed = urlparse(s)

        # Disallow credentialed URLs
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentialed URLs are not allowed")

        # Scheme validation
        scheme = (parsed.scheme or "").lower()
        if scheme and scheme not in ("http", "https"):
            raise ValueError("Only http and https schemes are allowed")

        host = parsed.hostname.lower().rstrip(".") if parsed.hostname else None

        # Absolute (or scheme-relative) URL handling
        if host:
            if not is_host_in_allowed_domains(host, VALID_REDIRECT_DOMAINS):
                raise ValueError("The domain is not an approved redirect target")

            # Determine scheme for scheme-relative URLs (e.g., //example.com)
            final_scheme = scheme or "https"

            # Normalize netloc: host + optional port (no userinfo)
            try:
                port = parsed.port  # can raise ValueError if invalid
            except ValueError:
                raise ValueError("Invalid port in URL")

            # IPv6 host normalization with brackets when reconstructing netloc
            netloc_host = host
            if ":" in netloc_host and not netloc_host.startswith("["):
                netloc_host = f"[{netloc_host}]"

            netloc = f"{netloc_host}:{port}" if port is not None else netloc_host

            path = parsed.path or "/"
            query = parsed.query
            fragment = parsed.fragment

            final_url = urlunparse((final_scheme, netloc, path, "", query, fragment))
            logger.info(
                "redirect_attempt result=success original=%r final=%r",
                original_input,
                final_url,
            )
            return final_url

        # Relative URL handling (no host)
        if scheme:
            # e.g., "http:example.com" (no //) is ambiguous -> reject
            raise ValueError("Invalid URL format: missing hostname")

        # Normalize to site-internal absolute path
        path = parsed.path or "/"
        if not path.startswith("/"):
            path = "/" + path

        query = parsed.query
        fragment = parsed.fragment

        final_url = urlunparse(("", "", path, "", query, fragment))
        logger.info(
            "redirect_attempt result=success original=%r final=%r",
            original_input,
            final_url,
        )
        return final_url

    except Exception as e:
        logger.warning(
            "redirect_attempt result=failure original=%r error=%s",
            original_input,
            f"{e.__class__.__name__}: {e}",
        )
        raise
