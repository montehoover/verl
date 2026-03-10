import logging
import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

logger = logging.getLogger(__name__)


def _extract_hostname(url: str):
    """
    Parse the URL and return a normalized (lowercase) hostname or None if not present.
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname.lower() if parsed.hostname else None
    logger.debug("Extracted hostname '%s' from URL '%s'", hostname, url)
    return hostname


def _validate_redirect_domain(redirect_url: str, allowed_domains) -> str:
    """
    Ensure the redirect_url's domain is included in the allowed_domains set.
    Returns the validated domain string.
    Raises ValueError if invalid.
    """
    logger.debug("Validating redirect_url domain for URL '%s'", redirect_url)
    domain = _extract_hostname(redirect_url)
    if domain and domain in allowed_domains:
        logger.debug("Redirect URL domain validated: '%s'", domain)
        return domain

    logger.error(
        "redirect_url domain validation failed: redirect_url='%s', domain='%s', allowed_domains=%s",
        redirect_url, domain, sorted(allowed_domains),
    )
    raise ValueError("redirect_url domain is not allowed")


def _build_query_string(existing_query: str, updates: dict) -> str:
    """
    Merge existing query parameters with updates and return a new query string.
    """
    query_params = dict(urllib.parse.parse_qsl(existing_query, keep_blank_values=True))
    query_params.update(updates)
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    logger.debug("Built query string with keys: %s", sorted(query_params.keys()))
    return new_query


def _rebuild_url_with_query(parsed_url: urllib.parse.ParseResult, query: str) -> str:
    """
    Return a URL string by replacing the query part of parsed_url with the provided query.
    """
    new_parsed = parsed_url._replace(query=query)
    final_url = urllib.parse.urlunparse(new_parsed)
    logger.debug("Rebuilt URL with new query; scheme=%s, netloc=%s, path=%s",
                 new_parsed.scheme, new_parsed.netloc, new_parsed.path)
    return final_url


def create_sso_url(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Construct an SSO redirect URL by combining the SSO endpoint with the user ID and a validated return URL.

    Raises:
        ValueError: If redirect_url's domain is not in ALLOWED_RETURN_DOMAINS.
    """
    logger.debug("Starting SSO URL creation for uid='%s'", uid)

    # Validate redirect URL domain
    _validate_redirect_domain(redirect_url, ALLOWED_RETURN_DOMAINS)

    # Parse the auth endpoint and prepare query parameters
    parsed_auth = urllib.parse.urlparse(auth_endpoint)
    logger.debug("Parsed auth endpoint: scheme=%s, netloc=%s, path=%s",
                 parsed_auth.scheme, parsed_auth.netloc, parsed_auth.path)

    new_query = _build_query_string(
        existing_query=parsed_auth.query,
        updates={"uid": uid, "redirect_url": redirect_url},
    )

    # Rebuild and return the final URL
    final_url = _rebuild_url_with_query(parsed_auth, new_query)
    logger.info("Constructed SSO URL for uid='%s'", uid)
    return final_url
