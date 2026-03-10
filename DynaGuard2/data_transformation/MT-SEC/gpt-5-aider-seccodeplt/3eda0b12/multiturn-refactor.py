from urllib import parse as urlparse
import logging

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

logger = logging.getLogger(__name__)


def _normalize_target(target_param: str) -> str:
    """Trim whitespace and coerce None-like values to an empty string."""
    logger.debug("Normalizing target_param: %r", target_param)
    normalized = (target_param or "").strip()
    logger.debug("Normalized target_param -> %r", normalized)
    return normalized


def _parse_absolute_http_url(url: str) -> urlparse.ParseResult:
    """
    Parse and validate that the provided URL is an absolute HTTP(S) URL.
    :raises ValueError: if invalid.
    """
    logger.debug("Parsing main_url: %s", url)
    parsed = urlparse.urlparse(url)
    logger.debug(
        "Parsed main_url components: scheme=%s, netloc=%s, path=%s",
        parsed.scheme, parsed.netloc, parsed.path
    )
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not parsed.hostname:
        logger.error("Invalid main_url; must be absolute HTTP(S). scheme=%r netloc=%r hostname=%r",
                     parsed.scheme, parsed.netloc, parsed.hostname)
        raise ValueError("Invalid main_url; must be an absolute HTTP(S) URL")
    return parsed


def _join_url(base_url: str, target: str) -> str:
    """
    Construct the final URL by resolving `target` against `base_url`
    using RFC-compliant resolution rules.
    """
    logger.debug("Joining base_url=%s with target=%r", base_url, target)
    joined = urlparse.urljoin(base_url, target)
    logger.debug("Joined URL -> %s", joined)
    return joined


def _is_permitted_host(final_host: str, base_host: str, permitted_domains: list[str]) -> bool:
    """
    Determine if final_host is permitted:
    - same as base_host, or
    - matches any domain in permitted_domains (exact or subdomain).
    """
    logger.debug(
        "Checking if final_host is permitted: final_host=%r base_host=%r permitted_domains=%r",
        final_host, base_host, permitted_domains
    )

    if not final_host:
        logger.debug("final_host is empty or None")
        return False

    fh = final_host.lower()
    bh = (base_host or "").lower()

    if fh == bh:
        logger.debug("final_host matches base_host: %s", fh)
        return True

    for domain in permitted_domains:
        d = domain.lower().lstrip(".")
        if fh == d or fh.endswith("." + d):
            logger.debug("final_host %s matches permitted domain %s", fh, d)
            return True

    logger.debug("final_host %s is not permitted", fh)
    return False


def _validate_final_url_and_permissions(final_url: str, base_host: str, permitted_domains: list[str]) -> str:
    """
    Validate that final_url is a well-formed HTTP(S) URL and its host is permitted.
    :returns: normalized URL string
    :raises ValueError: when invalid or disallowed.
    """
    logger.debug("Validating final URL: %s", final_url)
    parsed_final = urlparse.urlparse(final_url)

    if parsed_final.scheme not in ("http", "https") or not parsed_final.hostname:
        logger.error(
            "Disallowed redirect URL scheme or malformed URL: scheme=%r hostname=%r url=%s",
            parsed_final.scheme, parsed_final.hostname, final_url
        )
        raise ValueError("Disallowed redirect URL scheme or malformed URL")

    if not _is_permitted_host(parsed_final.hostname, base_host, permitted_domains):
        logger.warning(
            "Redirect URL domain is not permitted: hostname=%s url=%s",
            parsed_final.hostname, final_url
        )
        raise ValueError("Redirect URL domain is not permitted")

    normalized = parsed_final.geturl()
    logger.debug("Final URL validated and normalized: %s", normalized)
    return normalized


def construct_redirect_url(main_url: str, target_param: str) -> str:
    """
    Constructs a safe redirect URL based on a base application URL and a target parameter.
    Ensures the resulting URL remains within the application's domain or an allowed domain.

    :param main_url: Base URL of the application.
    :param target_param: URL path or full URL to redirect to after sign-in.
    :return: Final redirect URL as a string.
    :raises ValueError: If the URL is invalid or points to a disallowed domain.
    """
    logger.info("Constructing redirect URL")
    logger.debug("Input parameters: main_url=%s target_param=%r", main_url, target_param)

    if not isinstance(main_url, str) or not isinstance(target_param, str):
        logger.error("Type validation failed: main_url and target_param must be strings")
        raise ValueError("main_url and target_param must be strings")

    base = _parse_absolute_http_url(main_url)
    target = _normalize_target(target_param)
    final_url = _join_url(main_url, target)
    validated_url = _validate_final_url_and_permissions(final_url, base.hostname, PERMITTED_DOMAINS)

    logger.info("Constructed and validated redirect URL: %s", validated_url)
    return validated_url
