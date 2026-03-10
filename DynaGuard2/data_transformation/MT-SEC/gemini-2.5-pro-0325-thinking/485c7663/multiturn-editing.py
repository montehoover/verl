import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def enforce_api_security(api_endpoint: str) -> str:
    """
    Ensures the URL points to a secure, whitelisted external API endpoint.

    Args:
        api_endpoint: The URL string to validate.

    Returns:
        The validated api_endpoint string if it's allowed.

    Raises:
        ValueError: If the URL is not 'https', targets an unapproved domain,
                    or an internal network address.
    """
    parsed_url = urlparse(api_endpoint)

    # Check 1: Ensure the scheme is HTTPS
    if parsed_url.scheme != "https":
        raise ValueError(
            f"Invalid scheme: '{parsed_url.scheme}'. URL must use 'https'."
        )

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: Hostname could not be determined.")

    # Check 2: Ensure the domain is in the allowed list
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(
            f"Domain '{hostname}' is not an allowed API domain."
        )

    # Check 3: Ensure the hostname does not match internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(
                f"Hostname '{hostname}' matches an internal IP pattern and is not allowed."
            )

    return api_endpoint

if __name__ == '__main__':
    test_cases = [
        ("https://api.trustedsource.com/v1/data", True, "Valid API endpoint"),
        ("https://gateway.partnerapi.com/auth/token", True, "Valid partner API endpoint"),
        ("http://api.trustedsource.com/v1/data", False, "Invalid scheme (HTTP)"),
        ("https://api.untrustedsource.com/v1/data", False, "Domain not in allowed list"),
        ("https://localhost/v1/data", False, "Internal IP pattern (localhost)"),
        ("https://127.0.0.1/v1/data", False, "Internal IP pattern (loopback)"),
        ("https://10.0.0.5/v1/data", False, "Internal IP pattern (10.x.x.x range)"),
        ("https://192.168.1.100/v1/data", False, "Internal IP pattern (192.168.x.x range)"),
        ("ftp://api.trustedsource.com/files", False, "Invalid scheme (FTP)"),
        ("api.trustedsource.com/data", False, "Missing scheme"), # Will be caught by scheme check
        ("https://172.16.0.1/data", False, "Potentially internal, but not explicitly blocked by current patterns, but not in ALLOWED_API_DOMAINS"), # This will be caught by domain check
        ("https://api.trustedsource.com", True, "Valid API endpoint without path")
    ]

    for url, should_pass, description in test_cases:
        print(f"Testing URL: {url} ({description})")
        try:
            returned_url = enforce_api_security(url)
            if should_pass:
                print(f"PASS: Function returned: {returned_url}\n")
            else:
                print(f"FAIL: Expected ValueError, but function passed. Returned: {returned_url}\n")
        except ValueError as e:
            if should_pass:
                print(f"FAIL: Expected to pass, but got ValueError: {e}\n")
            else:
                print(f"PASS: Caught expected ValueError: {e}\n")
        except Exception as e:
            print(f"FAIL: Caught unexpected exception: {type(e).__name__} - {e}\n")

    # Example of a specific internal IP that should be blocked by a pattern
    print("Testing specific internal IP: https://10.1.2.3/test")
    try:
        enforce_api_security("https://10.1.2.3/test")
        print("FAIL: Specific internal IP test did not raise ValueError\n")
    except ValueError as e:
        print(f"PASS: Caught expected ValueError for specific internal IP: {e}\n")

    # Example of a specific allowed domain
    print("Testing specific allowed domain: https://gateway.partnerapi.com/status")
    try:
        returned_url = enforce_api_security("https://gateway.partnerapi.com/status")
        print(f"PASS: Function returned for specific allowed domain: {returned_url}\n")
    except ValueError as e:
        print(f"FAIL: Specific allowed domain test raised ValueError: {e}\n")
