import re
from urllib.parse import urlparse

def extract_protocol(url_string: str) -> str | None:
    """
    Extracts the protocol from a given URL string.

    Args:
        url_string: The URL string to parse.

    Returns:
        The protocol (e.g., 'http', 'https') if present, otherwise None.
    """
    parsed_url = urlparse(url_string)
    if parsed_url.scheme:
        return parsed_url.scheme
    return None


def parse_url_components(url_string: str) -> dict[str, str | None]:
    """
    Parses a URL string and extracts its protocol, domain, and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary with keys 'protocol', 'domain', and 'path'.
        Values can be None if a component is not present.
    """
    parsed_url = urlparse(url_string)
    return {
        "protocol": parsed_url.scheme if parsed_url.scheme else None,
        "domain": parsed_url.netloc if parsed_url.netloc else None,
        "path": parsed_url.path if parsed_url.path else None,
    }


def verify_path_format(address: str) -> bool:
    """
    Verifies if a given string is a valid http or https path using a regex.

    Args:
        address: The string address to validate.

    Returns:
        True if the path is a valid http or https URL format, False otherwise.
    """
    # Regex to check for http or https URLs.
    # It looks for:
    # ^           - start of the string
    # https?://   - http:// or https://
    # (?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+ # domain name part
    # [a-zA-Z]{2,6}                                    # TLD
    # (:\d+)?                                          # optional port
    # (/.*)?                                           # optional path
    # $           - end of the string
    # A simpler version could be used if strict domain validation isn't needed:
    # regex = r"^https?://[^\s/$.?#].[^\s]*$"
    regex = r"^https?://(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}(:\d+)?(/.*)?$"
    try:
        if re.match(regex, address):
            return True
        return False
    except Exception:
        # Should not happen with re.match and a string input, but as per requirement
        return False

if __name__ == '__main__':
    # Example Usage
    urls_to_test = [
        "http://www.example.com",
        "https://www.example.com/path?query=value",
        "ftp://ftp.example.com",
        "www.example.com",  # No protocol
        "://www.example.com", # Invalid protocol
        "example.com", # No protocol
        "http:/invalid-url", # Malformed but has scheme
        "", # Empty string
    ]

    for url in urls_to_test:
        protocol = extract_protocol(url)
        print(f"URL: '{url}', Protocol: {protocol}")

    # Test with a more complex case
    url_with_port = "https://localhost:8080/api/data"
    print(f"URL: '{url_with_port}', Protocol: {extract_protocol(url_with_port)}")

    print("\n--- Testing parse_url_components ---")
    urls_for_components_test = [
        "http://www.example.com/some/path?query=string#fragment",
        "https://api.example.co.uk:8080/v1/users?id=123",
        "ftp://ftp.fileserver.com/public/data.zip",
        "www.no-protocol.com/path",
        "/just/a/path",
        "http://domainonly.com",
        "https://domainwithslash.com/",
        "",
        "mailto:user@example.com" # scheme is mailto, no domain or path in http sense
    ]

    for url in urls_for_components_test:
        components = parse_url_components(url)
        print(f"URL: '{url}'")
        print(f"  Protocol: {components['protocol']}")
        print(f"  Domain:   {components['domain']}")
        print(f"  Path:     {components['path']}")
        print("-" * 20)

    print("\n--- Testing verify_path_format ---")
    paths_to_verify = [
        "http://www.example.com",
        "https://example.com/path/to/resource",
        "http://localhost:8080",
        "https://sub.domain.example.co.uk/api?query=123#section",
        "ftp://www.example.com", # Invalid protocol
        "www.example.com",       # Missing protocol
        "http//www.example.com", # Malformed protocol
        "https://",              # Incomplete
        "http://.com",           # Invalid domain
        "https://example .com",  # Space in domain
        "http://example.com:port", # Invalid port
        "https://example.com/a path with spaces", # Path with spaces (valid per regex if not considering URL encoding)
        "",                      # Empty string
        None,                    # None input (will be caught by try-except if re.match fails, or type hint error)
        12345                    # Integer input (will be caught by try-except)
    ]

    for path_to_verify in paths_to_verify:
        # Ensure path_to_verify is a string for re.match
        if not isinstance(path_to_verify, str):
            is_valid = False
            print(f"Path: '{path_to_verify}' (type: {type(path_to_verify).__name__}), Is Valid: {is_valid}")
        else:
            is_valid = verify_path_format(path_to_verify)
            print(f"Path: '{path_to_verify}', Is Valid: {is_valid}")
    print("-" * 20)
