import re
import ipaddress
from urllib.parse import urlsplit


_ALLOWED_SCHEMES = {"http", "https"}


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _validate_port(port_str: str) -> None:
    if not port_str.isdigit():
        raise ValueError("Port must be a number.")
    port = int(port_str)
    if port < 1 or port > 65535:
        raise ValueError("Port must be in the range 1-65535.")


def _validate_hostname(host: str) -> None:
    # Allow special-case "localhost"
    if host.lower() == "localhost":
        return

    # Remove trailing dot (FQDN notation)
    if host.endswith("."):
        host = host[:-1]

    if not host:
        raise ValueError("Hostname is empty.")

    # Convert IDN (Unicode) hostnames to ASCII using IDNA for validation
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise ValueError("Hostname contains invalid international characters.")

    if len(ascii_host) > 253:
        raise ValueError("Hostname exceeds the maximum length (253).")

    labels = ascii_host.split(".")
    label_re = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")

    for label in labels:
        if not label:
            raise ValueError("Hostname has an empty label.")
        if not label_re.match(label):
            raise ValueError(f"Invalid hostname label: {label}")


def _parse_hostport(netloc: str) -> tuple[str, str | None]:
    # Strip userinfo if present
    if "@" in netloc:
        _, hostport = netloc.rsplit("@", 1)
    else:
        hostport = netloc

    if hostport.startswith("["):
        # IPv6: [addr]:port?
        end = hostport.find("]")
        if end == -1:
            raise ValueError("Invalid IPv6 host: missing closing bracket.")
        host = hostport[1:end]
        rest = hostport[end + 1 :]
        port = None
        if rest:
            if rest.startswith(":"):
                port = rest[1:]
            else:
                raise ValueError("Invalid host: unexpected characters after IPv6 literal.")
        return host, port

    # IPv4 or hostname
    if ":" in hostport:
        host, port = hostport.rsplit(":", 1)
    else:
        host, port = hostport, None

    return host, port


def validate_url(url: str) -> bool:
    """
    Validate a URL string for safe parsing/use.

    Returns:
        True if the URL is valid.

    Raises:
        ValueError: if the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    url = url.strip()
    if not url:
        raise ValueError("URL must be a non-empty string.")

    # Disallow unencoded spaces or control characters anywhere in the URL
    if any(ord(ch) <= 0x1F or ord(ch) == 0x7F for ch in url) or " " in url:
        raise ValueError("URL contains spaces or control characters.")

    parts = urlsplit(url)

    if not parts.scheme:
        raise ValueError("URL must include a scheme (e.g., http or https).")

    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {scheme}. Allowed: {', '.join(sorted(_ALLOWED_SCHEMES))}.")

    if not parts.netloc:
        raise ValueError("URL must include a network location (host).")

    host, port_str = _parse_hostport(parts.netloc)

    # Validate host
    if not host:
        raise ValueError("Host is empty.")

    # Determine if host is IP (v4/v6) or a hostname
    is_ip = False
    try:
        # If brackets were present they were stripped in _parse_hostport
        ipaddress.IPv4Address(host)
        is_ip = True
    except ValueError:
        try:
            ipaddress.IPv6Address(host)
            is_ip = True
        except ValueError:
            is_ip = False

    if not is_ip:
        _validate_hostname(host)

    # Validate port if present
    if port_str is not None and port_str != "":
        _validate_port(port_str)

    # Optionally ensure path/query/fragment do not contain raw spaces or controls (already checked above)
    # Additional normalization or security checks could be added here if needed.

    return True
