from urllib.parse import urlparse, urlunparse
from typing import Dict

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def extract_url_components(url: str) -> Dict[str, str]:
    """
    Parse a URL and return its components.

    Returns a dict with keys: scheme, netloc, path, params, query, fragment.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }


def is_domain_allowed(url: str) -> bool:
    """
    Check if the URL's domain is in the allowed redirect domains list.
    Matches exact domains and their subdomains.
    """
    parsed = urlparse(url)

    # Handle URLs without scheme like "example.com/path"
    if not parsed.netloc and "://" not in url and not url.startswith("//"):
        parsed = urlparse(f"http://{url}")

    host = parsed.hostname
    if not host:
        return False

    host = host.rstrip(".").lower()
    try:
        host_idna = host.encode("idna").decode("ascii")
    except Exception:
        host_idna = host

    for allowed in ALLOWED_REDIRECT_DOMAINS:
        a = allowed.rstrip(".").lower()
        try:
            a_idna = a.encode("idna").decode("ascii")
        except Exception:
            a_idna = a

        if host_idna == a_idna or host_idna.endswith("." + a_idna):
            return True

    return False


def user_defined_redirect(url: str) -> str:
    """
    Accept a user-provided URL and return a sanitized URL suitable for
    secure redirection if the domain is allowed. Raises ValueError otherwise.
    """
    if not isinstance(url, str):
        raise ValueError("Invalid URL")

    raw = url.strip()
    if not raw or "\r" in raw or "\n" in raw:
        raise ValueError("Invalid URL")

    if not is_domain_allowed(raw):
        raise ValueError("Domain not allowed")

    parsed = urlparse(raw)
    if not parsed.netloc and "://" not in raw and not raw.startswith("//"):
        parsed = urlparse(f"http://{raw}")

    scheme = (parsed.scheme or "http").lower()
    if scheme not in ("http", "https"):
        raise ValueError("Invalid scheme")

    host = parsed.hostname
    if not host:
        raise ValueError("Invalid host")

    host = host.rstrip(".").lower()
    try:
        host_idna = host.encode("idna").decode("ascii")
    except Exception:
        raise ValueError("Invalid host")

    port = parsed.port
    if port is not None:
        if port < 1 or port > 65535:
            raise ValueError("Invalid port")
        netloc = f"{host_idna}:{port}"
    else:
        netloc = host_idna

    def clean(component: str) -> str:
        if not component:
            return ""
        return component.replace("\r", "").replace("\n", "")

    path = clean(parsed.path).replace("\\", "/")
    params = clean(parsed.params)
    query = clean(parsed.query)
    fragment = clean(parsed.fragment)

    if not path.startswith("/"):
        path = "/" + path if path else "/"

    sanitized_url = urlunparse((scheme, netloc, path, params, query, fragment))
    return sanitized_url
