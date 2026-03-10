from urllib.parse import urlsplit

def extract_path(url: str) -> str:
    """
    Extract the path component from a full URL, excluding query parameters and fragments.

    Examples:
    - https://example.com/foo/bar?x=1#y -> /foo/bar
    - https://example.com -> ""
    """
    parsed = urlsplit(url)
    return parsed.path
