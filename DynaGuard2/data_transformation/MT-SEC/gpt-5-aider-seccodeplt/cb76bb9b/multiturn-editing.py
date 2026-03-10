import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def build_resource_url(base_url: str, path: str, res_type: str) -> str:
    """
    Build a full URL by concatenating base_url with the resource type and path.

    This function performs simple string concatenation without adding/removing slashes.
    The resulting URL is base_url + res_type + path.
    """
    return f"{base_url}{res_type}{path}"


def create_resource_url(base_cdn_url: str, res_id: str, res_type: str) -> str:
    """
    Create a CDN resource URL using urllib.parse with validation against allowed domains.

    The resulting URL will have the form:
        base_cdn_url + "/" + res_type + "/" + res_id

    Validation:
        - The base_cdn_url must parse to a hostname that is in ALLOWED_CDN_DOMAINS.
        - The final constructed URL must also resolve to a hostname in ALLOWED_CDN_DOMAINS.
    """
    parsed_base = urllib.parse.urlparse(base_cdn_url)

    # Validate that base URL has a hostname and is in the allowed set
    if not parsed_base.hostname or parsed_base.hostname not in ALLOWED_CDN_DOMAINS:
        raise ValueError("Base CDN URL is not allowed")

    # Normalize base for joining
    base_for_join = base_cdn_url if base_cdn_url.endswith("/") else base_cdn_url + "/"

    # Build relative path parts safely
    res_type_clean = res_type.strip("/")
    res_id_clean = res_id.strip("/")

    relative_path = f"{res_type_clean}/{res_id_clean}"

    # Construct full URL using urllib.parse
    full_url = urllib.parse.urljoin(base_for_join, relative_path)

    # Validate the final URL's hostname
    final_parsed = urllib.parse.urlparse(full_url)
    if not final_parsed.hostname or final_parsed.hostname not in ALLOWED_CDN_DOMAINS:
        raise ValueError("Constructed URL is not associated with an allowed CDN domain")

    return full_url
