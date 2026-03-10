import urllib.parse

__all__ = ["validate_cdn_url", "generate_resource_path", "generate_external_resource_link"]

def _to_idna(host: str) -> str:
    try:
        return host.encode("idna").decode("ascii").lower()
    except Exception:
        return host.lower()

def _parse_hostname(url: str) -> str:
    """
    Parse the hostname from a URL string.
    Supports:
    - Full URLs (http/https)
    - Protocol-relative URLs (//example.com/path)
    - Bare hostnames (example.com/path)
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname

    if host is None and url.startswith("//"):
        # Protocol-relative URL
        parsed = urllib.parse.urlparse("https:" + url)
        host = parsed.hostname

    if host is None and "://" not in url:
        # Bare hostname without scheme
        parsed = urllib.parse.urlparse("https://" + url)
        host = parsed.hostname

    return host

# Predefined set of allowed CDN domains
ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def validate_cdn_url(url: str) -> str:
    """
    Validate that the given URL points to an allowed CDN domain.

    - Accepts full URLs (http/https), protocol-relative URLs (//host/path),
      and bare hostnames (host/path).
    - Returns the original URL if valid.
    - Raises ValueError if the URL is malformed, uses an invalid scheme,
      or the hostname is not in the allowed CDN domains list.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    # Normalize parsing for scheme-less URLs
    parsed_input = url
    if not url.startswith(("http://", "https://", "//")):
        parsed_input = "https://" + url

    parsed = urllib.parse.urlparse(parsed_input)
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

    host = _parse_hostname(url)
    if not host:
        raise ValueError("Invalid URL: hostname could not be determined")

    host_idna = _to_idna(host)

    # Exact or subdomain match on allowed CDN domains
    for allowed in ALLOWED_CDN_DOMAINS:
        if host_idna == allowed or host_idna.endswith("." + allowed):
            return url

    raise ValueError(f"Domain '{host}' is not in the list of allowed CDN domains")

def generate_resource_path(resource_tag: str, resource_category: str) -> str:
    """
    Generate a CDN resource path for a given tag and category.

    Supported categories:
    - 'image'  -> '/images/{tag}'
    - 'script' -> '/scripts/{tag}'

    The resource_tag may include sub-paths (e.g., 'icons/logo.png').
    Raises ValueError for invalid input or unsupported categories.
    """
    if not isinstance(resource_category, str) or not resource_category.strip():
        raise ValueError("resource_category must be a non-empty string")
    if not isinstance(resource_tag, str) or not resource_tag.strip():
        raise ValueError("resource_tag must be a non-empty string")

    category = resource_category.strip().lower()
    category_dirs = {
        "image": "images",
        "script": "scripts",
    }

    if category not in category_dirs:
        raise ValueError(f"Unsupported resource category: {resource_category}")

    # Normalize and validate the tag:
    # - Trim whitespace
    # - Use forward slashes
    # - Remove any leading slashes
    # - Disallow empty segments, '.' and '..'
    tag = resource_tag.strip().replace("\\", "/").lstrip("/")
    segments = tag.split("/")

    if not tag or any(seg in ("", ".", "..") for seg in segments):
        raise ValueError("Invalid resource_tag: must not contain empty, '.' or '..' segments")

    base_dir = category_dirs[category]
    return f"/{base_dir}/{tag}"

def generate_external_resource_link(cdn_host_url: str, resource_tag: str, resource_category: str) -> str:
    """
    Construct a full CDN URL for an external resource.

    Steps:
    - Validate the CDN host/domain is allowed.
    - Generate the resource-specific path.
    - Build and return the full URL (scheme + host + path).
    """
    # Validate CDN host/domain
    validate_cdn_url(cdn_host_url)

    # Parse CDN host, normalizing scheme-less inputs
    parsed_input = cdn_host_url
    if not cdn_host_url.startswith(("http://", "https://", "//")):
        parsed_input = "https://" + cdn_host_url

    parsed = urllib.parse.urlparse(parsed_input)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    if not netloc:
        # Fallback if somehow missing; derive from hostname parsing
        host = _parse_hostname(cdn_host_url)
        if not host:
            raise ValueError("Invalid CDN host URL")
        netloc = host

    # Generate the resource path
    resource_path = generate_resource_path(resource_tag, resource_category)

    # Construct the final URL; ignore any path/query/fragment on the base host
    return urllib.parse.urlunparse((scheme, netloc, resource_path, "", "", ""))
