import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def generate_external_resource_link(cdn_host_url: str, resource_tag: str, resource_category: str) -> str:
    """
    Constructs a URL to access resources hosted on a CDN, ensuring the domain is allowed.

    Args:
        cdn_host_url: The base CDN's URL (e.g., "https://cdn.example.com").
                      Must include a scheme (http or https).
        resource_tag: A unique identifier for the resource (e.g., "image.jpg").
        resource_category: The category of the resource (e.g., "images", "scripts").

    Returns:
        The complete URL as a string (e.g., "https://cdn.example.com/images/image.jpg").

    Raises:
        ValueError: If the cdn_host_url's domain is not in ALLOWED_CDN_DOMAINS,
                    or if cdn_host_url is malformed (e.g., missing scheme or domain).
    """
    parsed_url = urllib.parse.urlparse(cdn_host_url)

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(
            f"Invalid CDN host URL: '{cdn_host_url}'. Must include scheme (e.g., 'https://') and domain."
        )

    if parsed_url.netloc not in ALLOWED_CDN_DOMAINS:
        raise ValueError(
            f"Domain '{parsed_url.netloc}' from URL '{cdn_host_url}' is not an allowed CDN domain."
        )

    # Path construction logic (modifying arguments for conciseness, as in previous version)
    # Ensure cdn_host_url for concatenation does not end with a slash
    if cdn_host_url.endswith('/'):
        cdn_host_url = cdn_host_url[:-1]
    
    # Ensure resource_category and resource_tag do not start with a slash
    if resource_category.startswith('/'):
        resource_category = resource_category[1:]
    if resource_tag.startswith('/'):
        resource_tag = resource_tag[1:]

    return f"{cdn_host_url}/{resource_category}/{resource_tag}"
