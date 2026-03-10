from urllib import parse as urlparse
import logging

logger = logging.getLogger(__name__)

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _validate_base_url(app_url: str) -> urlparse.ParseResult:
    """
    Parse and validate the base application URL.
    Raises ValueError if invalid.
    """
    logger.debug("Validating base app_url: %s", app_url)
    base = urlparse.urlparse(app_url)
    if not base.scheme or not base.netloc:
        logger.error("Invalid app_url detected: scheme=%s, netloc=%s", base.scheme, base.netloc)
        raise ValueError("Invalid app_url: must be an absolute URL with scheme and host")
    logger.debug("Base URL validated: scheme=%s, host=%s", base.scheme, base.hostname)
    return base


def _construct_candidate_url(base: urlparse.ParseResult, app_url: str, redirect_param: str) -> str:
    """
    Construct the candidate absolute URL to redirect to, based on the base app URL
    and the user-provided redirect parameter. This function is pure and does not
    perform domain validation.
    """
    logger.debug("Constructing candidate URL from app_url=%s redirect_param=%s", app_url, redirect_param)

    if not redirect_param:
        logger.debug("No redirect_param provided; using app_url as destination.")
        return app_url

    dest = urlparse.urlparse(redirect_param)

    if dest.scheme and dest.netloc:
        logger.debug("redirect_param is an absolute URL. Using it directly.")
        return redirect_param
    elif redirect_param.startswith("//"):
        candidate = f"{base.scheme}:{redirect_param}"
        logger.debug("redirect_param is protocol-relative. Constructed candidate: %s", candidate)
        return candidate
    else:
        candidate = urlparse.urljoin(app_url, redirect_param)
        logger.debug("redirect_param is relative. Joined to base -> candidate: %s", candidate)
        return candidate


def _is_allowed_host(host: str, base_host: str) -> bool:
    """
    Determine if the host is allowed: either matches the application's host or
    is present in the approved domains list (case-insensitive).
    """
    if not host:
        return False
    normalized_host = host.lower()
    normalized_base = (base_host or "").lower()
    approved = {d.lower() for d in APPROVED_DOMAINS}
    allowed = normalized_host == normalized_base or normalized_host in approved
    logger.debug(
        "Allowed host check: host=%s base_host=%s approved=%s => allowed=%s",
        normalized_host, normalized_base, sorted(approved), allowed
    )
    return allowed


def _validate_final_destination(base: urlparse.ParseResult, candidate_url: str) -> urlparse.ParseResult:
    """
    Validate the final redirect destination:
    - Scheme must be http or https
    - Host must exist
    - Host must match application host or be in approved domains
    Raises ValueError if invalid.
    """
    logger.debug("Validating final destination URL: %s", candidate_url)
    final = urlparse.urlparse(candidate_url)

    if final.scheme not in ("http", "https"):
        logger.error("Unsupported scheme in redirect URL: %s", final.scheme)
        raise ValueError("Invalid redirect URL: unsupported scheme")

    host = (final.hostname or "").lower()
    if not host:
        logger.error("Missing host in redirect URL.")
        raise ValueError("Invalid redirect URL: missing host")

    base_host = (base.hostname or "").lower()
    if not _is_allowed_host(host, base_host):
        logger.error("Redirect to unapproved domain attempted: %s", host)
        raise ValueError("Redirect to unapproved domain is not allowed")

    logger.debug(
        "Final destination validated: scheme=%s host=%s path=%s query=%s",
        final.scheme, final.hostname, final.path, final.query
    )
    return final


def generate_redirect_link(app_url: str, redirect_param: str) -> str:
    """
    Generates a safe redirect URL after successful login.

    Args:
        app_url: Base URL of the application (e.g., "https://myapp.example.com").
        redirect_param: A URL path (e.g., "/dashboard") or a full URL.

    Returns:
        The final redirect URL as an absolute string.

    Raises:
        ValueError: If the final URL is invalid or not within the application's domain
                    or one of the approved external domains.
    """
    logger.info("Generating redirect link: app_url=%s redirect_param=%s", app_url, redirect_param)
    base = _validate_base_url(app_url)
    candidate = _construct_candidate_url(base, app_url, redirect_param)
    final = _validate_final_destination(base, candidate)
    final_url = urlparse.urlunparse(final)
    logger.info("Generated redirect URL: %s", final_url)
    return final_url
