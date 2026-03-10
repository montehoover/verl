import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$",      # Private 192.168.x.x range
    # Add other private ranges if necessary, e.g., 172.16.0.0/12
    r"^172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}$" # Private 172.16.0.0 - 172.31.255.255
]

def validate_api_endpoint(api_url):
    """
    Ensures a given URL points to an external trusted API endpoint.

    Args:
        api_url: The URL string to validate.

    Returns:
        The validated URL string if it meets the criteria.

    Raises:
        ValueError: If the URL is invalid, not 'https', points to an internal
                    IP, or is not in the allowed domains.
    """
    parsed_url = urlparse(api_url)

    # 1. Check for a valid scheme (must be https)
    if parsed_url.scheme != 'https':
        raise ValueError("Invalid URL scheme: Only 'https' is allowed.")

    # 2. Check for a hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: Hostname is missing.")

    # 3. Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Invalid URL: Hostname '{hostname}' matches an internal IP pattern.")

    # 4. Check if the hostname is a direct IP address (and not an allowed domain)
    #    This is a stricter check. If an IP is used, it must resolve to an allowed domain,
    #    but simple string matching for IPs is prone to bypass (e.g. decimal/octal IPs).
    #    For now, we'll rely on the domain check. A more robust solution would involve
    #    DNS resolution, but that adds complexity and runtime dependencies.
    #    A simple check: if it looks like an IP and isn't an internal one (already checked),
    #    it might be an external IP. We'll allow this if it resolves to an allowed domain,
    #    which is covered by the next check.

    # 5. Check if the hostname is in the list of allowed API domains
    #    This check should handle subdomains correctly if needed, e.g. by checking
    #    `.endswith()` for each domain in ALLOWED_API_DOMAINS.
    #    For exact match as per current ALLOWED_API_DOMAINS:
    if hostname not in ALLOWED_API_DOMAINS:
        # Allow subdomains of the allowed domains
        is_allowed_subdomain = False
        for allowed_domain in ALLOWED_API_DOMAINS:
            if hostname.endswith("." + allowed_domain) or hostname == allowed_domain:
                is_allowed_subdomain = True
                break
        if not is_allowed_subdomain:
            raise ValueError(f"Invalid URL: Hostname '{hostname}' is not an allowed API domain.")

    return api_url


if __name__ == '__main__':
    test_urls = {
        "Valid URL (exact match)": "https://api.trustedsource.com/v1/data",
        "Valid URL (subdomain)": "https://user.api.trustedsource.com/v1/data",
        "Valid URL (partner)": "https://gateway.partnerapi.com/auth/token",
        "Invalid Scheme (http)": "http://api.trustedsource.com/v1/data",
        "Invalid Domain": "https://api.untrusted.com/v1/data",
        "Internal IP (localhost)": "https://localhost/api/data",
        "Internal IP (127.0.0.1)": "https://127.0.0.1/api/data",
        "Internal IP (10.x.x.x)": "https://10.0.0.5/api/data",
        "Internal IP (192.168.x.x)": "https://192.168.1.100/api/data",
        "Internal IP (172.16.x.x)": "https://172.16.0.10/api/data",
        "No Hostname": "https:///path/only",
        "External IP (not in allowed domains)": "https://8.8.8.8/some/api" # This should fail domain check
    }

    print("--- API Endpoint Validation ---")
    for description, url in test_urls.items():
        print(f"\nTesting URL: {url} ({description})")
        try:
            validated_url = validate_api_endpoint(url)
            print(f"  Result: Valid -> {validated_url}")
        except ValueError as e:
            print(f"  Result: Invalid -> {e}")
        print("-" * 30)
