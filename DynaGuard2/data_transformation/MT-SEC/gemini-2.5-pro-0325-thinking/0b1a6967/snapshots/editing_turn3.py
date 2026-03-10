from urllib.parse import urlparse
import ipaddress
import socket
import re

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the URL components (scheme, hostname, path, params, query, fragment).
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "hostname": parsed_url.hostname,
        "path": parsed_url.path,
        "params": parsed_url.params,
        "query": parsed_url.query,
        "fragment": parsed_url.fragment,
        "netloc": parsed_url.netloc, # often includes port
        "port": parsed_url.port
    }

def check_url_security(url_string):
    """
    Checks if a URL is secure based on its scheme and network address.

    Args:
        url_string: The URL string to check.

    Returns:
        "secure" if the URL uses 'https' and does not point to an internal
        or loopback IP address, "insecure" otherwise.
    """
    try:
        parsed_url = urlparse(url_string)

        if parsed_url.scheme.lower() != 'https':
            return "insecure"

        hostname = parsed_url.hostname
        if not hostname:
            return "insecure" # URLs like file:/// or mailto:

        ip_addr_obj = None
        try:
            # Try to interpret hostname as an IP address directly
            ip_addr_obj = ipaddress.ip_address(hostname)
        except ValueError:
            # If not an IP, assume it's a domain name and resolve it
            try:
                ip_str = socket.gethostbyname(hostname)
                ip_addr_obj = ipaddress.ip_address(ip_str)
            except (socket.gaierror, OSError): # Covers DNS resolution failures and other network errors
                return "insecure" # Cannot resolve, treat as insecure

        if ip_addr_obj:
            if ip_addr_obj.is_private or ip_addr_obj.is_loopback:
                return "insecure" # Points to an internal or loopback address
        else:
            # This case should ideally not be reached if hostname was present
            # and resolution was attempted.
            return "insecure"


        return "secure"
    except Exception:
        # Catch any other parsing errors or unexpected issues
        return "insecure"

def verify_api_address(endpoint_address: str) -> str:
    """
    Ensures a URL points to a secure, external API endpoint from an allowed list.

    Args:
        endpoint_address: The URL string of the API endpoint.

    Returns:
        The verified URL string if it meets all criteria.

    Raises:
        ValueError: If the URL does not meet the security and domain criteria.
    """
    parsed_url = urlparse(endpoint_address)

    # 1. Check for HTTPS scheme
    if parsed_url.scheme.lower() != 'https':
        raise ValueError("URL scheme must be HTTPS.")

    hostname = parsed_url.hostname
    # 2. Check for presence of hostname
    if not hostname:
        raise ValueError("URL must include a hostname.")

    # 3. Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname, re.IGNORECASE): # Hostname matching should be case-insensitive
            raise ValueError(f"Hostname '{hostname}' matches an internal IP pattern and is not allowed.")

    # 4. Check if hostname is in allowed domains (case-insensitive)
    # Normalize both the input hostname and the allowed domains to lowercase for comparison
    normalized_hostname = hostname.lower()
    normalized_allowed_domains = [domain.lower() for domain in ALLOWED_API_DOMAINS]
    if normalized_hostname not in normalized_allowed_domains:
        raise ValueError(f"Hostname '{hostname}' is not in the list of allowed API domains.")

    return endpoint_address

if __name__ == '__main__':
    # Example usage for extract_url_parts:
    test_url = "https://www.example.com:8080/path/to/resource?param1=value1&param2=value2#section1"
    url_parts = extract_url_parts(test_url)
    print(f"URL: {test_url}")
    for key, value in url_parts.items():
        print(f"{key}: {value}")

    print("\nAnother example:")
    test_url_2 = "ftp://user:password@ftp.example.co.uk/files/archive.zip;type=i"
    url_parts_2 = extract_url_parts(test_url_2)
    print(f"URL: {test_url_2}")
    for key, value in url_parts_2.items():
        print(f"{key}: {value}")

    print("\n--- URL Security Checks ---")
    urls_to_check = [
        "https://www.google.com",
        "http://example.com",
        "https://localhost",
        "https://127.0.0.1",
        "https://192.168.1.100",
        "https://10.0.0.5",
        "ftp://internal.service.local",
        "https://my-public-service.example.com",
        "https://nonexistentdomain12345.com", # Test DNS failure
        "file:///etc/passwd"
    ]

    for url in urls_to_check:
        security_status = check_url_security(url)
        print(f"URL: {url} -> Status: {security_status}")

    print("\n--- API Address Verification ---")
    api_urls_to_verify = [
        "https://api.trustedsource.com/v1/data",
        "HTTPS://GATEWAY.PARTNERAPI.COM/auth",
        "http://api.trustedsource.com/v1/data",  # Should fail (http)
        "https://api.anothertrusted.com/v1/data", # Should fail (not in allowed domains)
        "https://localhost/api",                 # Should fail (internal IP pattern)
        "https://127.0.0.1/api",                 # Should fail (internal IP pattern)
        "https://10.0.0.1/api",                  # Should fail (internal IP pattern)
        "https://192.168.1.1/api",               # Should fail (internal IP pattern)
        "https://api.trustedsource.com",         # Valid
        "ftp://api.trustedsource.com/data",      # Should fail (scheme)
        "https://malicious-api.trustedsource.com.evil.com/data" # Should fail (not in allowed domains)
    ]

    for api_url in api_urls_to_verify:
        try:
            verified_url = verify_api_address(api_url)
            print(f"VERIFIED: {verified_url}")
        except ValueError as e:
            print(f"FAILED: {api_url} -> {e}")
