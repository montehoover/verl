from urllib import parse as urlparse
from ipaddress import ip_address
import re

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def parse_and_validate_url(url: str) -> dict:
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    url = url.strip()
    if not url:
        raise ValueError("URL is empty")

    parsed = urlparse.urlparse(url)

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


def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenate a relative URL path to a validated base URL.

    - base_url must be an absolute HTTP/HTTPS URL.
    - path must be a relative path (may include query/fragment), not a full URL.
    """
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a string")
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    base_url = base_url.strip()
    path = path.strip()

    if not base_url:
        raise ValueError("base_url is empty")

    # Validate the base URL (ensures http/https and valid domain).
    parse_and_validate_url(base_url)

    # Disallow absolute URLs as the path to prevent overriding the base.
    parsed_path = urlparse.urlparse(path)
    if parsed_path.scheme or parsed_path.netloc:
        raise ValueError("path must be a relative URL path, not a full URL")

    # Ensure proper joining semantics for "directory-like" bases.
    base_for_join = base_url
    if not base_for_join.endswith("/") and not path.startswith("/"):
        base_for_join += "/"

    return urlparse.urljoin(base_for_join, path)


def generate_redirect_link(app_url: str, redirect_param: str) -> str:
    """
    Generate a safe redirect URL based on an application base URL and a user-provided redirect parameter.

    - app_url: absolute base URL of the application.
    - redirect_param: may be a relative path or an absolute URL.
    Returns the full URL as a string if it targets the application's domain or an approved domain.
    Raises ValueError otherwise.
    """
    if not isinstance(app_url, str):
        raise ValueError("app_url must be a string")
    if not isinstance(redirect_param, str):
        raise ValueError("redirect_param must be a string")

    app_url = app_url.strip()
    redirect_param = redirect_param.strip()

    # Validate and extract app domain (ensures http/https and valid domain)
    app_info = parse_and_validate_url(app_url)
    app_domain = app_info["domain"].lower()

    # Build the final URL using standard URL resolution rules.
    # Ensure proper joining for base that doesn't end with "/" when redirect_param is relative.
    base_for_join = app_url
    if not base_for_join.endswith("/") and not redirect_param.startswith("/"):
        base_for_join += "/"
    final_url = urlparse.urljoin(base_for_join, redirect_param or "/")

    # Parse the final URL and validate scheme and host.
    final_parsed = urlparse.urlparse(final_url)
    final_scheme = (final_parsed.scheme or "").lower()
    if final_scheme not in ("http", "https"):
        raise ValueError("Redirect URL must use http or https")

    final_host = (final_parsed.hostname or "").lower()
    if not final_host:
        raise ValueError("Redirect URL must include a host")

    # Normalize hosts to IDNA for comparison
    def to_idna(host: str) -> str:
        try:
            return host.encode("idna").decode("ascii").lower()
        except Exception:
            return host.lower()

    app_domain_idna = to_idna(app_domain)
    approved_domains_idna = [to_idna(d) for d in APPROVED_DOMAINS]
    final_host_idna = to_idna(final_host)

    def is_subdomain_or_equal(host: str, domain: str) -> bool:
        return host == domain or host.endswith("." + domain)

    allowed = (
        is_subdomain_or_equal(final_host_idna, app_domain_idna)
        or any(is_subdomain_or_equal(final_host_idna, d) for d in approved_domains_idna)
    )

    if not allowed:
        raise ValueError("Redirect target is not allowed")

    return final_url
