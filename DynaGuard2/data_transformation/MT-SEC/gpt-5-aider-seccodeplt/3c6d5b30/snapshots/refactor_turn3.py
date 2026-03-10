from urllib import parse as urlparse
import logging

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def construct_redirect_link(domain_base_url: str, next_redirect_param: str) -> str:
    """
    Construct a safe redirect URL for post-login redirection.

    Args:
        domain_base_url: Absolute base URL of the application (e.g., "https://app.example.com/login").
        next_redirect_param: The path or URL the user should be sent to after login.

    Returns:
        A string with domain_base_url augmented with a `next` query parameter.

    Raises:
        ValueError: If domain_base_url is not absolute, if next_redirect_param uses a disallowed scheme,
                    or if next_redirect_param points to a host neither within the base host nor in ALLOWED_DOMAINS.
    """
    logger.debug(
        "Starting construct_redirect_link with base_url=%s, next_param=%r",
        domain_base_url, next_redirect_param
    )

    try:
        base_parts, base_host = _parse_base_url(domain_base_url)
    except ValueError as e:
        logger.error("Invalid domain_base_url: %s", e)
        raise
    logger.debug(
        "Parsed base URL: scheme=%s, netloc=%s, path=%s, host=%s",
        base_parts.scheme, base_parts.netloc, base_parts.path, base_host
    )

    next_value = _normalize_next(next_redirect_param)
    logger.debug("Normalized next parameter: %r", next_value)

    next_parts = _parse_next(next_value)
    logger.debug(
        "Parsed next parameter: scheme=%s, netloc=%s, path=%s, hostname=%s",
        next_parts.scheme or "", next_parts.netloc or "", next_parts.path or "", next_parts.hostname or ""
    )

    try:
        _validate_next_scheme(next_parts)
    except ValueError as e:
        logger.warning(
            "Rejected next parameter due to disallowed scheme: %r (error: %s)",
            next_parts.scheme, e
        )
        raise
    else:
        logger.debug("Next parameter scheme validated: %r", next_parts.scheme or "(none)")

    target_host = next_parts.hostname
    if not target_host:
        logger.debug("Next target is a relative URL; considered within base domain")
        # No host validation needed for relative paths
        pass
    else:
        same = _is_same_host(target_host, base_host)
        allowed_ext = _is_in_allowed_domains(target_host, ALLOWED_DOMAINS)
        logger.debug(
            "Target host evaluation: base_host=%s, target_host=%s, same_host=%s, allowed_external=%s",
            base_host, target_host, same, allowed_ext
        )
        try:
            _validate_target_host(next_parts, base_host, ALLOWED_DOMAINS)
        except ValueError as e:
            logger.warning(
                "Rejected next parameter due to unauthorized host: %s (error: %s)",
                target_host, e
            )
            raise
        else:
            logger.debug("Next parameter host validated: %s", target_host)

    try:
        result = _build_result_url(base_parts, next_value)
    except Exception as e:
        logger.exception("Failed building result URL: %s", e)
        raise

    logger.info("Constructed redirect URL: %s", result)
    return result


# ----- Pipeline steps (pure functions) -----

def _parse_base_url(domain_base_url: str):
    """
    Validate and parse the base URL. Returns (SplitResult, base_host).
    """
    if not isinstance(domain_base_url, str) or not domain_base_url.strip():
        raise ValueError("domain_base_url must be a non-empty string")

    base_parts = urlparse.urlsplit(domain_base_url)
    if not base_parts.scheme or not base_parts.netloc:
        raise ValueError("domain_base_url must be an absolute URL with scheme and host")

    base_host = base_parts.hostname or ""
    return base_parts, base_host


def _normalize_next(next_redirect_param) -> str:
    """
    Normalize the next value to a string (allow empty string).
    """
    return "" if next_redirect_param is None else str(next_redirect_param)


def _parse_next(next_value: str):
    """
    Parse the next redirect value. Returns SplitResult.
    """
    return urlparse.urlsplit(next_value)


def _validate_next_scheme(next_parts) -> None:
    """
    Disallow dangerous or non-HTTP(S) schemes in the `next` target.
    """
    if next_parts.scheme and next_parts.scheme.lower() not in ("http", "https"):
        raise ValueError("next_redirect_param uses a disallowed URL scheme")


def _validate_target_host(next_parts, base_host: str, allowed_domains) -> None:
    """
    Validate that the target host (if provided) is either the same as base_host
    or within the allowed external domains.
    """
    target_host = next_parts.hostname  # None for relative paths
    if target_host:
        if not _is_same_host(target_host, base_host) and not _is_in_allowed_domains(target_host, allowed_domains):
            raise ValueError("Redirection target is not within the base domain or allowed external domains")


def _build_result_url(base_parts, next_value: str) -> str:
    """
    Build the resulting URL with an updated `next` query parameter.
    """
    existing_qs = urlparse.parse_qsl(base_parts.query, keep_blank_values=True)
    # Remove existing 'next' parameters if present
    existing_qs = [(k, v) for (k, v) in existing_qs if k.lower() != "next"]
    existing_qs.append(("next", next_value))

    new_query = urlparse.urlencode(existing_qs, doseq=True)
    return urlparse.urlunsplit((
        base_parts.scheme,
        base_parts.netloc,
        base_parts.path,
        new_query,
        base_parts.fragment,
    ))


# ----- Shared helpers -----

def _is_same_host(candidate: str, base_host: str) -> bool:
    """
    Returns True if candidate host exactly matches the base host.
    """
    return (candidate or "").lower() == (base_host or "").lower()


def _is_in_allowed_domains(host: str, allowed_domains=None) -> bool:
    """
    Returns True if host matches one of the allowed domains or their subdomains.
    """
    if not host:
        return False
    host = host.lower().rstrip(".")
    domains = allowed_domains if allowed_domains is not None else ALLOWED_DOMAINS
    for allowed in domains:
        allowed = allowed.lower().rstrip(".")
        if host == allowed or host.endswith("." + allowed):
            return True
    return False
