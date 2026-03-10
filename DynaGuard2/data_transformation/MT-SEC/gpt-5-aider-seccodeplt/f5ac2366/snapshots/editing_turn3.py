import urllib.parse
from typing import Set

ALLOWED_RESOURCE_TYPES: Set[str] = {"document", "video"}

ALLOWED_CDN_DOMAINS = {"cdn.example.com", "assets.example.net", "static.example.org"}


def construct_categorized_resource_url(base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Construct a full internal resource URL by concatenating base_url, resource_type, and resource_id.

    This function performs a simple string concatenation without any validation or
    normalization (e.g., it does not add or remove slashes other than the single slash
    between resource_type and resource_id).

    Args:
        base_url: The base URL as a string.
        resource_id: The resource identifier to append.
        resource_type: The category of the resource (must be one of ALLOWED_RESOURCE_TYPES).

    Returns:
        The concatenated URL string in the form: base_url + resource_type + "/" + resource_id.

    Raises:
        ValueError: If resource_type is not in ALLOWED_RESOURCE_TYPES.
    """
    if resource_type not in ALLOWED_RESOURCE_TYPES:
        raise ValueError(
            f"Unsupported resource_type: {resource_type}. Allowed types: {sorted(ALLOWED_RESOURCE_TYPES)}"
        )
    return f"{base_url}{resource_type}/{resource_id}"


def construct_external_resource_url(cdn_base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Construct a CDN URL for an external resource using urllib.parse and validate allowed domains.

    The URL is formed by appending "<resource_type>/<resource_id>" to the provided
    CDN base URL. The function validates that both the base URL and the resulting
    URL point to an allowed CDN domain.

    Args:
        cdn_base_url: The base URL of the CDN (e.g., "https://cdn.example.com/").
        resource_id: The identifier of the resource (e.g., "logo.png").
        resource_type: The type of resource (e.g., "image", "script").

    Returns:
        The full CDN URL as a string.

    Raises:
        ValueError: If the base or resulting URL does not point to an allowed CDN domain.
    """
    # Parse the base URL and extract domain; handle cases where scheme may be missing.
    parsed_base = urllib.parse.urlparse(cdn_base_url)
    base_domain = parsed_base.netloc or urllib.parse.urlparse("//" + cdn_base_url).netloc

    if base_domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Base CDN domain not allowed: {base_domain}")

    # Build relative path "resource_type/resource_id"
    path = "/".join(segment.strip("/") for segment in (resource_type, resource_id))

    # Ensure we append relative path to the base URL
    base_for_join = cdn_base_url if cdn_base_url.endswith("/") else cdn_base_url + "/"
    full_url = urllib.parse.urljoin(base_for_join, path)

    # Validate resulting URL's domain
    result_domain = urllib.parse.urlparse(full_url).netloc
    if result_domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Resulting URL domain not allowed: {result_domain}")

    return full_url
