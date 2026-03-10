from urllib.parse import urlparse
from ipaddress import ip_address
import re


def parse_and_validate_url(url: str) -> dict:
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    url = url.strip()
    if not url:
        raise ValueError("URL is empty")

    parsed = urlparse(url)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")

    if not parsed.netloc:
        raise ValueError("URL must include a domain")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid domain")

    # Validate hostname as either an IP address or a valid DNS name (including IDNs).
    def is_valid_hostname(host: str) -> bool:
        if host == "localhost":
            return True
        # IP address check
        try:
            ip_address(host)
            return True
        except ValueError:
            pass

        # IDN compatibility: encode using IDNA; if this fails, it's invalid.
        try:
            host_idna = host.encode("idna").decode("ascii")
        except Exception:
            return False

        if len(host_idna) > 253:
            return False

        labels = host_idna.split(".")
        for label in labels:
            if not label or len(label) > 63:
                return False
            # Must start and end with alphanumeric, may contain hyphens in the middle.
            if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label):
                return False
        return True

    if not is_valid_hostname(hostname):
        raise ValueError("URL has an invalid domain")

    result = {
        "scheme": scheme,
        "domain": hostname,
        "path": parsed.path or "/",
    }

    if parsed.port is not None:
        result["port"] = parsed.port
    if parsed.query:
        result["query"] = parsed.query
    if parsed.fragment:
        result["fragment"] = parsed.fragment
    if parsed.username:
        result["username"] = parsed.username
    if parsed.password:
        result["password"] = parsed.password

    return result
