from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    """
    Append or update image-processing query parameters on the provided image URL.
    Validates that the URL scheme is supported (http or https) before modification.

    Parameters:
        img_url (str): The original image URL.

    Returns:
        str: The URL with appended parameters for resizing and formatting,
             ensuring size=medium and format=jpg are present.

    Raises:
        ValueError: If the URL scheme is unsupported or missing.
    """
    parsed = urlparse(img_url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("Unsupported or missing URL scheme. Allowed schemes are: http, https.")

    query = parse_qs(parsed.query, keep_blank_values=True)

    # Ensure desired parameters are present/updated
    query["size"] = ["medium"]
    query["format"] = ["jpg"]

    new_query = urlencode(query, doseq=True)
    updated = parsed._replace(query=new_query)
    return urlunparse(updated)
