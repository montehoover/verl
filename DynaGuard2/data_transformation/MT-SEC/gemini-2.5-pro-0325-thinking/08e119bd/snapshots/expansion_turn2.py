from urllib.parse import urlparse

ALLOWED_CDN_DOMAINS = [
    "cdn.example.com",
    "assets.example-cdn.net",
    "static.another-cdn.org",
]

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
