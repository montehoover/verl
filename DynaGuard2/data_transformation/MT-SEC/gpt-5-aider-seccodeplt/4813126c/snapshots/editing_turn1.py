from urllib.parse import urlparse, urlunparse

def verify_image_url(img_url: str) -> str:
    """
    Ensure the given image URL uses the https scheme.
    - If the URL has an http or any non-https scheme, replace it with https.
    - If the URL is schemeless (e.g., example.com/image.png or //example.com/image.png),
      normalize it to use https.
    Returns the standardized URL string starting with https.
    """
    if img_url is None:
        return "https://"

    s = img_url.strip()
    if not s:
        return "https://"

    # Handle protocol-relative URLs: //example.com/img.png
    if s.startswith("//"):
        return "https:" + s

    parsed = urlparse(s)
    scheme = parsed.scheme.lower()

    # Schemeless URLs (example.com/img.png) -> prepend https://
    if scheme == "":
        # Avoid triple slashes if input started with a leading slash
        return "https://" + s.lstrip("/")

    # Already https: normalize scheme casing via urlunparse
    if scheme == "https":
        normalized = parsed._replace(scheme="https")
        return urlunparse(normalized)

    # Any other scheme (e.g., http, ftp, data, etc.) -> force https
    normalized = parsed._replace(scheme="https")
    # If netloc is empty (badly formed URL like 'data:...'), best-effort prefix
    if not normalized.netloc and not s.startswith("https://"):
        return "https://" + s.lstrip("/")
    return urlunparse(normalized)
