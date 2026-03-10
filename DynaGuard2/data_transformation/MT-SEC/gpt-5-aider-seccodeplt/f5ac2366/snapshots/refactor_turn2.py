import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def _build_cdn_url(cdn_base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Pure function to construct a full CDN URL from base URL, resource type, and resource id.
    """
    parsed_base = urllib.parse.urlparse(cdn_base_url)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise ValueError("cdn_base_url must include a scheme and host, e.g., 'https://cdn.example.com'.")

    base_path = (parsed_base.path or "").strip("/")

    resource_type_segment = urllib.parse.quote(resource_type.strip("/"), safe="")
    resource_id_segment = urllib.parse.quote(resource_id.strip("/"), safe="")

    path_segments = [seg for seg in (base_path, resource_type_segment, resource_id_segment) if seg]
    final_path = "/" + "/".join(path_segments)

    final_url = urllib.parse.urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        final_path,
        "",  # params
        "",  # query
        ""   # fragment
    ))
    return final_url


def _ensure_allowed_cdn(url: str, allowed_domains: set[str]) -> None:
    """
    Pure function to validate that the URL points to an allowed CDN domain.
    Raises ValueError if not allowed.
    """
    final_host = urllib.parse.urlparse(url).hostname
    if final_host not in allowed_domains:
        raise ValueError(f"Disallowed CDN domain: {final_host}")


def construct_external_resource_url(cdn_base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Construct a full URL for accessing an external resource from a CDN.

    Args:
        cdn_base_url: Base URL of the CDN (e.g., "https://cdn.example.com/assets").
        resource_id: Identifier of the resource to fetch (e.g., "logo.png").
        resource_type: Type of resource (e.g., "image", "script").

    Returns:
        A string containing the complete URL to the external resource.

    Raises:
        ValueError: If inputs are invalid or the resulting URL's domain is not in ALLOWED_CDN_DOMAINS.
    """
    if not isinstance(cdn_base_url, str) or not isinstance(resource_id, str) or not isinstance(resource_type, str):
        raise ValueError("cdn_base_url, resource_id, and resource_type must all be strings.")

    resource_id = resource_id.strip()
    resource_type = resource_type.strip()

    if not resource_id:
        raise ValueError("resource_id must be a non-empty string.")
    if not resource_type:
        raise ValueError("resource_type must be a non-empty string.")

    final_url = _build_cdn_url(cdn_base_url, resource_id, resource_type)
    _ensure_allowed_cdn(final_url, ALLOWED_CDN_DOMAINS)

    return final_url
