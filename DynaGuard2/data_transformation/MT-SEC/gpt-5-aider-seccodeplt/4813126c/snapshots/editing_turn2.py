from urllib.parse import urlparse
import requests

ALLOWED_SCHEMES = ["http", "https"]

def verify_image_url(img_url: str) -> str:
    """
    Ensure the URL uses https and verify the host is reachable via an HTTP HEAD request.
    - Accepts http/https URLs; any other scheme raises ValueError.
    - Schemeless or protocol-relative URLs are normalized to https.
    - Performs a HEAD request to the resulting https URL.
    - Raises ValueError if unreachable; returns the https URL otherwise.
    """
    if not isinstance(img_url, str):
        raise ValueError("img_url must be a string.")

    s = img_url.strip()
    if not s:
        raise ValueError("img_url cannot be empty.")

    # Handle protocol-relative URLs: //example.com/img.png
    if s.startswith("//"):
        s = "https:" + s

    parsed = urlparse(s)

    # Schemeless URLs (example.com/img.png) -> prepend https://
    if not parsed.scheme:
        s = "https://" + s.lstrip("/")
        parsed = urlparse(s)

    scheme_lower = parsed.scheme.lower()
    if scheme_lower not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    # Force https scheme
    parsed_https = parsed._replace(scheme="https")
    https_url = parsed_https.geturl()

    # Ensure a host is present
    if not parsed_https.netloc:
        raise ValueError("URL must include a host.")

    # Check reachability using a HEAD request
    try:
        requests.head(https_url, allow_redirects=True, timeout=(3.05, 5))
    except requests.RequestException as exc:
        raise ValueError(f"URL is unreachable: {exc}") from exc

    return https_url
