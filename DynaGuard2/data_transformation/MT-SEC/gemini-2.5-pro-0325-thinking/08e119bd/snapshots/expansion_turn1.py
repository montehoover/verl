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

if __name__ == '__main__':
    # Example Usage
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
