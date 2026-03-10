import urllib.parse
from typing import Set

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def _is_allowed_cdn_domain(hostname: str, allowed_domains: Set[str]) -> bool:
    """
    Check if the given hostname is within the allowed CDN domains.

    Args:
        hostname: The hostname extracted from the base CDN URL.
        allowed_domains: A set of allowed CDN domain names.

    Returns:
        True if the hostname is allowed, False otherwise.
    """
    return bool(hostname) and hostname in allowed_domains


def _construct_resource_url(parsed_base: urllib.parse.ParseResult, res_type: str, res_id: str) -> str:
    """
    Construct the final resource URL by combining the parsed base URL with encoded path segments.

    Args:
        parsed_base: The parsed result of the base CDN URL.
        res_type: The kind of resource (e.g., "image", "script").
        res_id: The unique identifier of the resource.

    Returns:
        The fully constructed resource URL as a string.
    """
    # Safely encode path components
    type_component = urllib.parse.quote(res_type or "", safe="-._~")
    id_component = urllib.parse.quote(res_id or "", safe="-._~")

    # Ensure base path ends with a slash before appending segments
    base_path = parsed_base.path or ""
    if not base_path.endswith("/"):
        base_path = (base_path + "/") if base_path else "/"

    # Construct the new path
    new_path = f"{base_path}{type_component}/{id_component}"

    # Reassemble the URL, clearing params/query/fragment for a clean resource URL
    final_parts = parsed_base._replace(path=new_path, params="", query="", fragment="")
    return urllib.parse.urlunparse(final_parts)


def create_resource_url(base_cdn_url: str, res_id: str, res_type: str) -> str:
    """
    Build a CDN resource URL from a base CDN URL, a resource identifier, and a resource type.

    Args:
        base_cdn_url: The base URL of the CDN (e.g., "https://cdn.example.com/assets/").
        res_id: The unique identifier of the resource.
        res_type: The kind of resource (e.g., "image", "script").

    Returns:
        A string containing the complete URL for the resource on the CDN.

    Raises:
        ValueError: If the constructed URL is not associated with any allowed CDN domains.
    """
    # Parse and validate the base CDN URL host against allowed domains
    parsed = urllib.parse.urlparse(base_cdn_url.strip() if base_cdn_url is not None else "")
    hostname = parsed.hostname
    if not _is_allowed_cdn_domain(hostname, ALLOWED_CDN_DOMAINS):
        raise ValueError("Constructed URL is not associated with any allowed CDN domains.")

    # Build and return the resource URL
    return _construct_resource_url(parsed, res_type, res_id)
