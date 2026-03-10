import re

_HOST_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOST_PATTERN = rf"(?:{_HOST_LABEL}\.)+{_HOST_LABEL}"
_URL_RE = re.compile(
    rf"^(?:https?)://(?:[^@/?#]+@)?(?P<host>{_HOST_PATTERN})(?::\d+)?(?:[/?#]|$)",
    re.IGNORECASE,
)

def fetch_email_domain(url: str):
    """
    Extract the full host (including subdomains), e.g., 'sub.example.com', from a well-formed HTTP/HTTPS URL.
    Returns None if the URL is not well-formed or does not contain a valid domain.
    """
    if not isinstance(url, str):
        return None

    url = url.strip()
    match = _URL_RE.match(url)
    if not match:
        return None

    host = match.group("host").lower()
    return host
