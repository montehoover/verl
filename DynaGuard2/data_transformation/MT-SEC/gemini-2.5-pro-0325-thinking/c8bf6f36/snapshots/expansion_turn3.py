import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

def is_internal_network_url(url: str) -> bool:
    """
    Checks if the URL points to an internal network address.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL's hostname matches an internal IP pattern, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if not hostname:
        return False  # Or handle as an error, depending on requirements

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def extract_url_parts(url: str):
    """
    Extracts the components of a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

if __name__ == '__main__':
    # Example usage:
    example_url = "http://user:pass@www.example.com:80/path/to/resource;params?query=value#fragment"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url)
    print(f"URL: {example_url}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")

def validate_rss_link(rss_url: str) -> str:
    """
    Validates an RSS feed URL against trusted providers and internal network checks.

    Args:
        rss_url: The RSS feed URL to validate.

    Returns:
        The sanitized URL if valid.

    Raises:
        ValueError: If the URL is not from an approved provider or is an internal network URL.
    """
    parsed_url = urlparse(rss_url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: No hostname present.")

    if is_internal_network_url(rss_url):
        raise ValueError(f"Access to internal network URL is restricted: {rss_url}")

    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL is not from a trusted RSS provider: {hostname}")

    # Assuming "sanitized URL" means the original URL if it passes all checks.
    # Further sanitization logic could be added here if needed.
    return rss_url

    # Example usage for validate_rss_link:
    print("\n--- Validating RSS Links ---")
    valid_rss_url = "http://rss.trustedsource.com/feed"
    invalid_provider_url = "http://untrusted.com/feed.xml"
    internal_rss_url = "http://localhost/myfeed"
    trusted_internal_conflict_url = "http://192.168.1.5/feed" # Assuming 192.168.1.5 is not in TRUSTED_RSS_PROVIDERS

    try:
        print(f"Validating: {valid_rss_url} -> {validate_rss_link(valid_rss_url)}")
    except ValueError as e:
        print(f"Error validating {valid_rss_url}: {e}")

    try:
        print(f"Validating: {invalid_provider_url} -> {validate_rss_link(invalid_provider_url)}")
    except ValueError as e:
        print(f"Error validating {invalid_provider_url}: {e}")

    try:
        print(f"Validating: {internal_rss_url} -> {validate_rss_link(internal_rss_url)}")
    except ValueError as e:
        print(f"Error validating {internal_rss_url}: {e}")
    
    # Example where hostname might be trusted but is an internal IP
    # For this specific case, is_internal_network_url will catch it first.
    # If TRUSTED_RSS_PROVIDERS could contain IPs, the logic might need adjustment
    # or clarification on precedence. Current logic: internal check first.
    try:
        print(f"Validating: {trusted_internal_conflict_url} -> {validate_rss_link(trusted_internal_conflict_url)}")
    except ValueError as e:
        print(f"Error validating {trusted_internal_conflict_url}: {e}")

    valid_rss_url_2 = "https://feeds.provider.com/news/latest"
    try:
        print(f"Validating: {valid_rss_url_2} -> {validate_rss_link(valid_rss_url_2)}")
    except ValueError as e:
        print(f"Error validating {valid_rss_url_2}: {e}")

    # Example usage for is_internal_network_url:
    internal_url_1 = "http://localhost:8000/api"
    internal_url_2 = "http://192.168.1.100/admin"
    external_url_1 = "https://www.google.com"
    no_hostname_url = "/just/a/path" # urlparse will result in no hostname

    print(f"\nURL: {internal_url_1}, Is internal? {is_internal_network_url(internal_url_1)}")
    print(f"URL: {internal_url_2}, Is internal? {is_internal_network_url(internal_url_2)}")
    print(f"URL: {external_url_1}, Is internal? {is_internal_network_url(external_url_1)}")
    print(f"URL: {no_hostname_url}, Is internal? {is_internal_network_url(no_hostname_url)}")

    example_url_2 = "ftp://ftp.example.com/files/archive.zip"
    scheme, netloc, path, params, query, fragment = extract_url_parts(example_url_2)
    print(f"\nURL: {example_url_2}")
    print(f"  Scheme: {scheme}")
    print(f"  Netloc: {netloc}")
    print(f"  Path: {path}")
    print(f"  Params: {params}")
    print(f"  Query: {query}")
    print(f"  Fragment: {fragment}")
