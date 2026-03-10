from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    """
    Validates an image URL, confirming its domain is part of a trusted list
    and its scheme is supported.

    Args:
        img_url: The input image URL.

    Returns:
        The resolved valid URL if it belongs to the list of trusted domains
        and has an allowed scheme.

    Raises:
        ValueError: If the domain is not in the trusted list, the scheme is
                    unsupported, or the URL is malformed.
    """
    try:
        parsed_url = urlparse(img_url)
    except Exception as e: # Catch potential errors from urlparse itself, though rare for typical strings
        raise ValueError(f"Invalid URL format: {img_url}") from e

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Malformed URL: {img_url}. Scheme and domain must be present.")

    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    if parsed_url.netloc.lower() not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(
            f"Untrusted domain: '{parsed_url.netloc}'. "
            f"Allowed domains are: {', '.join(TRUSTED_IMAGE_HOSTS)}."
        )

    return img_url

if __name__ == '__main__':
    # Example Usage:
    test_urls = [
        "https://img.example.com/image123.png",
        "http://cdn.trusted.com/path/to/image.jpg",
        "https://images.hosting.com/another/image.gif",
        "ftp://untrusted.com/image.bmp",  # Invalid scheme
        "https://other.example.com/pic.png",  # Untrusted domain
        "img.example.com/missing_scheme.png", # Malformed URL
        "//img.example.com/schemeless.png" # Malformed URL (for this validator)
    ]

    for url in test_urls:
        try:
            valid_url = validate_image_url(url)
            print(f"Validated: {valid_url}")
        except ValueError as e:
            print(f"Validation failed for {url}: {e}")

    print("\nTesting with provided example:")
    example_input = {"img_url": "https://img.example.com/image123.png"}
    try:
        output = validate_image_url(example_input["img_url"])
        print(f"Input: {example_input['img_url']}, Output: {output}")
    except ValueError as e:
        print(f"Error for {example_input['img_url']}: {e}")
