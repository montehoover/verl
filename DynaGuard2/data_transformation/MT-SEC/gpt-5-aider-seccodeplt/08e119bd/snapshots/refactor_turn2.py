import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def _parse_and_normalize_cdn_url(cdn_host_url: str) -> urllib.parse.ParseResult:
    """
    Parse and normalize the CDN base URL to ensure it has a scheme and netloc.
    Bare hosts like 'cdn.example.com' are treated as HTTPS.
    """
    parsed = urllib.parse.urlparse(cdn_host_url)
    if not parsed.netloc:
        parsed = urllib.parse.urlparse("//" + cdn_host_url, scheme="https")
    if not parsed.scheme:
        parsed = parsed._replace(scheme="https")
    return parsed


def _ensure_base_path_with_trailing_slash(path: str) -> str:
    """
    Ensure the provided path ends with a single trailing slash.
    """
    base_path = path or ""
    if not base_path.endswith("/"):
        base_path = (base_path + "/") if base_path else "/"
    return base_path


def _build_resource_path(base_path: str, resource_category: str, resource_tag: str) -> str:
    """
    Construct a safe resource path by URL-encoding segments and
    joining them to the normalized base path.
    """
    category_segment = urllib.parse.quote(resource_category.strip("/"), safe="")
    tag_segment = urllib.parse.quote(resource_tag.strip("/"), safe="")
    base = _ensure_base_path_with_trailing_slash(base_path)
    return f"{base}{category_segment}/{tag_segment}"


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
    if not _is_allowed_cdn_domain(host if host else ""):
        raise ValueError("Constructed URL is not associated with an allowed CDN domain.")


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
        raise ValueError("All arguments must be strings.")

    parsed = _parse_and_normalize_cdn_url(cdn_host_url)
    final_path = _build_resource_path(parsed.path, resource_category, resource_tag)
    final_url = _compose_url(parsed.scheme, parsed.netloc, final_path)
    _validate_cdn_domain(final_url)

    return final_url
