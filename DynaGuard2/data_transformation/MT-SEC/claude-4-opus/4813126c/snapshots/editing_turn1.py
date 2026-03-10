from urllib.parse import urlparse, urlunparse

def verify_image_url(img_url: str) -> str:
    parsed = urlparse(img_url)
    https_parsed = parsed._replace(scheme='https')
    return urlunparse(https_parsed)
