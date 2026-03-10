from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse


__all__ = ["extract_url_parts", "check_internal_access", "validate_callback"]

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string into its components.

    Returns a dictionary with:
      - scheme
      - netloc
      - path
      - params
      - query
      - fragment
      - username
      - password
      - hostname
      - port
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    candidate = url.strip()

    # Initial parse
    parsed = urlparse(candidate)

    # Handle schemeless URLs like "example.com/path"
    if (
        not parsed.scheme
        and not parsed.netloc
        and "://" not in candidate
        and not candidate.startswith("/")
        and candidate
    ):
        parsed = urlparse(f"//{candidate}")

    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
    }


def check_internal_access(url: str) -> Optional[bool]:
    """
    Return True if the URL targets internal/private addresses such as:
      - localhost
      - 127.0.0.1
      - 10.0.0.0/8 (detected via '10.' prefix)
      - 192.168.0.0/16 (detected via '192.168.' prefix)

    Returns:
      - True if the URL is considered internal/private according to the above rules.
      - None if the URL is considered safe (not matching the internal rules).
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parts = extract_url_parts(url)
    host = parts.get("hostname")
    if not host:
        return None

    host_l = host.lower().rstrip(".")

    for token in INTERNAL_IP_RANGES:
        if token.endswith("."):
            if host_l.startswith(token):
                return True
        else:
            if host_l == token:
                return True

    return None


def validate_callback(url: str) -> str:
    """
    Validate a webhook callback URL:
      - Must not target internal/private networks.
      - Must originate from an approved domain.
      - Must use http(s); schemeless inputs default to https.
      - Username/password components are stripped in the returned URL.
      - Returns a normalized, safe URL string or raises ValueError.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parts = extract_url_parts(url)
    host = parts.get("hostname")
    if not host:
        raise ValueError("Invalid URL: missing hostname")

    # Deny internal/private targets
    if check_internal_access(url):
        raise ValueError("URL points to an internal/private network")

    # Enforce approved domains (exact host match)
    host_l = host.lower().rstrip(".")
    if host_l not in {d.lower() for d in APPROVED_API_DOMAINS}:
        raise ValueError("Unapproved callback domain")

    # Enforce/normalize scheme
    scheme = (parts.get("scheme") or "").lower()
    if scheme == "":
        scheme = "https"
    elif scheme not in ("http", "https"):
        raise ValueError("Unsupported URL scheme")

    # Rebuild a safe netloc without credentials
    hostname = host_l
    port = parts.get("port")
    netloc = f"{hostname}:{port}" if port else hostname

    # Normalize path
    path = parts.get("path") or "/"
    if not path.startswith("/"):
        path = "/" + path.lstrip("/")

    # Preserve params and query; drop fragment
    params = parts.get("params") or ""
    query = parts.get("query") or ""
    fragment = ""

    safe_url = urlunparse((scheme, netloc, path, params, query, fragment))
    return safe_url
