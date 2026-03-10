import re

def extract_url_components(url):
    """
    Extracts the protocol and domain from a URL string using regular expressions.

    Args:
        url: The URL string.

    Returns:
        A tuple (protocol, domain) if the URL is structured correctly, 
        otherwise (None, None).
    """
    # Regex to capture protocol (optional) and domain
    # It looks for a protocol (e.g., http, https, ftp) followed by ://
    # Then it captures the domain name, which can include subdomains,
    # and excludes paths, query parameters, or fragments.
    # It also handles URLs without a protocol (e.g., www.example.com)
    regex = r"^(?:([a-zA-Z]+)://)?([^/?#:]+)(?:[/?#:]|$)"
    match = re.match(regex, url)
    
    if match:
        protocol = match.group(1)
        domain = match.group(2)
        return protocol, domain
    else:
        # Fallback for cases like "example.com/path" where protocol is missing
        # and the initial regex might not catch it if it expects "://"
        # This simpler regex just tries to get the first part before a slash, query, or hash
        # if no protocol was found.
        simple_domain_regex = r"^([^/?#:]+)(?:[/?#:]|$)"
        simple_match = re.match(simple_domain_regex, url)
        if simple_match:
            return None, simple_match.group(1)
        return None, None

if __name__ == '__main__':
    # Example Usage
    urls = [
        "http://www.example.com/path/to/page",
        "https://example.com",
        "ftp://ftp.example.co.uk/resource",
        "www.another-example.com/query?param=value",
        "example.com/page",
        "http://localhost:8080/api",
        "https://sub.domain.example.com",
        "invalid-url",
        "example.net", # URL with no protocol, no path
        "//example.com/path" # Protocol-relative URL
    ]

    for url_string in urls:
        protocol, domain = extract_url_components(url_string)
        print(f"URL: {url_string}, Protocol: {protocol}, Domain: {domain}")

    # Test cases
    print("\nAdditional Test Cases:")
    test_urls = {
        "http://example.com/path": ("http", "example.com"),
        "https://www.example.org": ("https", "www.example.org"),
        "example.net/resource": (None, "example.net"),
        "ftp://user:pass@example.com:21/file": ("ftp", "user:pass@example.com"), # Domain includes user:pass
        "localhost:3000": (None, "localhost"),
        "http://127.0.0.1/test": ("http", "127.0.0.1"),
        "example.com": (None, "example.com")
    }

    for url_string, expected in test_urls.items():
        protocol, domain = extract_url_components(url_string)
        print(f"URL: {url_string}, Got: ({protocol}, {domain}), Expected: {expected}")
        assert (protocol, domain) == expected, f"Test failed for {url_string}"
    
    print("\nAll additional tests passed.")
