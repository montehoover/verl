import logging
import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

logger = logging.getLogger(__name__)


def _parse_and_normalize_cdn_url(cdn_host_url: str) -> urllib.parse.ParseResult:
    """
    Parse and normalize the CDN base URL to ensure it has a scheme and netloc.
    Bare hosts like 'cdn.example.com' are treated as HTTPS.
    """
    parsed = urllib.parse.urlparse(cdn_host_url)
    if not parsed.netloc:
        logger.debug("CDN URL missing netloc; treating as bare host: %s", cdn_host_url)
        parsed = urllib.parse.urlparse("//" + cdn_host_url, scheme="https")

    if not parsed.scheme:
        logger.debug("CDN URL missing scheme; defaulting to https for host: %s", parsed.netloc)
        parsed = parsed._replace(scheme="https")

    logger.debug(
        "Normalized CDN URL -> scheme: %s, host: %s, path: %s",
        parsed.scheme, parsed.netloc, parsed.path or "/"
    )
    return parsed


def _ensure_base_path_with_trailing_slash(path: str) -> str:
    """
    Ensure the provided path ends with a single trailing slash.
    """
    base_path = f"{(path or '').rstrip('/')}/"
    return base_path


def _build_resource_path(base_path: str, resource_category: str, resource_tag: str) -> str:
    """
    Construct a safe resource path by URL-encoding segments and
    joining them to the normalized base path.
    """
    category_segment = urllib.parse.quote(resource_category.strip("/"), safe="")
    tag_segment = urllib.parse.quote(resource_tag.strip("/"), safe="")
    base = _ensure_base_path_with_trailing_slash(base_path)
    final_path = f"{base}{category_segment}/{tag_segment}"
    logger.debug(
        "Built resource path -> base: %s, category: %s, tag: %s, final_path: %s",
        base, category_segment, tag_segment, final_path
    )
    return final_path


def _compose_url(scheme: str, netloc: str, path: str) -> str:
    """
    Compose a full URL from scheme, netloc, and path without params, query, or fragment.
    """
    return urllib.parse.urlunparse((scheme, netloc, path, "", "", ""))


def _is_allowed_cdn_domain(host: str) -> bool:
    """
    Check if the given host belongs to the allowed CDN domains.
    """
    return bool(host) and host.lower() in ALLOWED_CDN_DOMAINS


def _validate_cdn_domain(final_url: str) -> None:
    """
    Validate that the final URL's host is in the allowed CDN domains.
    """
    host = urllib.parse.urlparse(final_url).hostname
    if not _is_allowed_cdn_domain(host or ""):
        logger.error(
            "CDN domain validation failed -> url: %s, host: %s, allowed_domains: %s",
            final_url, host, ", ".join(sorted(ALLOWED_CDN_DOMAINS))
        )
        raise ValueError("Constructed URL is not associated with an allowed CDN domain.")
    logger.debug("CDN domain validated -> host: %s", host)


def generate_external_resource_link(cdn_host_url: str, resource_tag: str, resource_category: str) -> str:
    """
    Builds a complete URL for a resource hosted on a CDN.

    Args:
        cdn_host_url: The base URL of the CDN (e.g., 'https://cdn.example.com').
        resource_tag: The unique identifier of the resource.
        resource_category: The kind of resource (e.g., 'image', 'script').

    Returns:
        A string that forms the complete URL for the resource on the CDN.

    Raises:
        ValueError: If the constructed URL is not associated with any allowed CDN domains.
    """
    if not isinstance(cdn_host_url, str) or not isinstance(resource_tag, str) or not isinstance(resource_category, str):
        logger.error(
            "Invalid argument types -> cdn_host_url: %r, resource_tag: %r, resource_category: %r",
            type(cdn_host_url).__name__, type(resource_tag).__name__, type(resource_category).__name__
        )
        raise ValueError("All arguments must be strings.")

    logger.debug(
        "Generating external resource link -> base_url: %s, category: %s, tag: %s",
        cdn_host_url, resource_category, resource_tag
    )

    parsed = _parse_and_normalize_cdn_url(cdn_host_url)
    final_path = _build_resource_path(parsed.path, resource_category, resource_tag)
    final_url = _compose_url(parsed.scheme, parsed.netloc, final_path)

    logger.info("Generated CDN URL: %s", final_url)

    _validate_cdn_domain(final_url)

    return final_url
