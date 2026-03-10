import urllib.parse # Already present, but good to note for context
from urllib.parse import urlparse, urljoin

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def validate_cdn_url(url: str) -> None:
    """
    Validates if the given URL belongs to an allowed CDN domain.

    Args:
        url: The URL string to validate.

    Raises:
        ValueError: If the URL's domain is not in the allowed list
                    or if the URL is malformed.
    """
    if not isinstance(url, str):
        raise TypeError("URL must be a string.")

    try:
        parsed_url = urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {url}. Error: {e}")

    domain = parsed_url.netloc
    if not domain:
        raise ValueError(f"Could not extract domain from URL: {url}")

    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(
            f"Domain '{domain}' is not an allowed CDN domain. "
            f"URL: {url}"
        )

def generate_resource_path(resource_tag: str, resource_category: str) -> str:
    """
    Generates a CDN resource path based on its tag and category.

    Args:
        resource_tag: The unique tag for the resource (e.g., "logo.png", "main.js").
        resource_category: The category of the resource (e.g., "image", "script").

    Returns:
        The resource path string.

    Raises:
        TypeError: If resource_tag or resource_category are not strings.
        ValueError: If the resource_category is not supported.
    """
    if not isinstance(resource_tag, str) or not isinstance(resource_category, str):
        raise TypeError("Resource tag and category must be strings.")

    if resource_category == "image":
        return f"/images/{resource_tag}"
    elif resource_category == "script":
        return f"/js/{resource_tag}"
    else:
        raise ValueError(f"Unsupported resource category: {resource_category}")

def generate_external_resource_link(cdn_host_url: str, resource_tag: str, resource_category: str) -> str:
    """
    Constructs a full URL for an external resource hosted on a CDN.

    Args:
        cdn_host_url: The base URL of the CDN host (e.g., "https://cdn.example.com").
        resource_tag: The unique tag for the resource.
        resource_category: The category of the resource.

    Returns:
        The full URL string for the resource.

    Raises:
        ValueError: If the cdn_host_url is not an allowed CDN or is malformed,
                    or if the resource_category is not supported.
        TypeError: If any of the arguments are not strings.
    """
    if not all(isinstance(arg, str) for arg in [cdn_host_url, resource_tag, resource_category]):
        raise TypeError("All arguments (cdn_host_url, resource_tag, resource_category) must be strings.")

    validate_cdn_url(cdn_host_url)  # Validates the domain
    resource_path = generate_resource_path(resource_tag, resource_category)

    # Ensure cdn_host_url ends with a slash for proper joining if it's just a domain
    # However, urljoin handles this well.
    # If cdn_host_url might not have a scheme, urlparse().scheme can check.
    # For simplicity, assuming cdn_host_url is a well-formed base URL (e.g., "https://cdn.example.com")
    
    # urljoin handles cases where cdn_host_url might or might not have a trailing slash,
    # and resource_path starts with a slash.
    full_url = urljoin(cdn_host_url, resource_path)
    return full_url

if __name__ == '__main__':
    # Example Usage for validate_cdn_url
    print("--- Validating CDN URLs ---")
    test_urls = [
        "https://cdn.example.com/path/to/resource.js",
        "http://assets.example-cdn.net/styles/main.css",
        "https://static.another-cdn.org/images/logo.png",
        "ftp://cdn.example.com/data.zip", # Scheme doesn't matter for this validation
        "https://www.regular-website.com/page",
        "cdn.example.com/only/path", # Malformed, no scheme
        "invalid-url",
        "http://unknown-cdn.com/file.txt"
    ]

    print(f"Allowed CDN domains: {ALLOWED_CDN_DOMAINS}\n")

    for test_url in test_urls:
        try:
            validate_cdn_url(test_url)
            print(f"OK: {test_url}")
        except (ValueError, TypeError) as e:
            print(f"Error: {e}")
        print("-" * 20)

    # Example with a non-string input
    try:
        validate_cdn_url(123)
    except TypeError as e:
        print(f"Error with non-string input: {e}")
    print("-" * 20)

    # Example Usage for generate_resource_path
    print("\n--- Generating Resource Paths ---")
    test_resources = [
        ("logo.png", "image"),
        ("main.js", "script"),
        ("style.css", "stylesheet"), # Unsupported category
        ("data.json", "image"), # Valid category, different tag
        (123, "script"), # Invalid tag type
        ("app.js", 456), # Invalid category type
    ]

    for tag, category in test_resources:
        try:
            path = generate_resource_path(tag, category)
            print(f"Generated path for ({tag}, {category}): {path}")
        except (ValueError, TypeError) as e:
            print(f"Error generating path for ({tag}, {category}): {e}")
        print("-" * 20)

    # Example Usage for generate_external_resource_link
    print("\n--- Generating External Resource Links ---")
    # Update ALLOWED_CDN_DOMAINS for these tests to match the new set
    # This is already done at the top of the file.
    # The validate_cdn_url function will use the updated global ALLOWED_CDN_DOMAINS.

    test_external_resources = [
        ("https://cdn.example.com", "logo.svg", "image"),
        ("http://assets.example.net/", "bundle.js", "script"),
        ("https://static.example.org", "photo.jpg", "image"),
        ("https://unallowed-cdn.com", "exploit.js", "script"), # Unallowed domain
        ("cdn.example.com", "icon.png", "image"), # Malformed cdn_host_url (no scheme)
        ("https://cdn.example.com", "archive.zip", "archive"), # Unsupported category
        (123, "main.js", "script"), # Invalid cdn_host_url type
        ("https://cdn.example.com", 456, "image"), # Invalid resource_tag type
        ("https://cdn.example.com", "app.js", 789), # Invalid resource_category type
    ]

    for host, tag, category in test_external_resources:
        try:
            link = generate_external_resource_link(host, tag, category)
            print(f"Generated link for ({host}, {tag}, {category}): {link}")
        except (ValueError, TypeError) as e:
            print(f"Error generating link for ({host}, {tag}, {category}): {e}")
        print("-" * 20)
