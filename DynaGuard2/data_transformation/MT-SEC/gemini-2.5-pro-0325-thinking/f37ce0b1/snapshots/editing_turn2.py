import re

def fetch_email_domain(url_string: str) -> str | None:
    """
    Extracts the full domain (including subdomains) from an HTTP/HTTPS URL.

    Args:
        url_string: The URL to parse.

    Returns:
        The full domain (e.g., "sub.example.com") or None if the URL is
        not a valid HTTP/HTTPS URL or no domain is found.
    """
    # Regex to capture the hostname (domain and subdomains) from http or https URLs.
    # It looks for "http://" or "https://"
    # then captures a sequence of characters that are not a forward slash or whitespace.
    match = re.search(r"https?://([^/\s]+)", url_string)
    if match:
        # The first captured group (index 1) is the hostname.
        return match.group(1)
    return None

if __name__ == '__main__':
    # Example Usage
    urls_to_test = [
        "https://sub.example.com/page",
        "http://example.com/another/page?query=param",
        "https://localhost:8080/path",
        "http://127.0.0.1/index.html",
        "ftp://ftp.example.com/resource",  # Should return None
        "example.com",  # Should return None
        "not a valid url",  # Should return None
        "https://another-example.co.uk/somepath",
        "http://test-domain.info",
    ]

    for url in urls_to_test:
        domain = fetch_email_domain(url)
        print(f"URL: \"{url}\" -> Domain: \"{domain}\"")

    # Expected output:
    # URL: "https://sub.example.com/page" -> Domain: "sub.example.com"
    # URL: "http://example.com/another/page?query=param" -> Domain: "example.com"
    # URL: "https://localhost:8080/path" -> Domain: "localhost:8080"
    # URL: "http://127.0.0.1/index.html" -> Domain: "127.0.0.1"
    # URL: "ftp://ftp.example.com/resource" -> Domain: "None"
    # URL: "example.com" -> Domain: "None"
    # URL: "not a valid url" -> Domain: "None"
    # URL: "https://another-example.co.uk/somepath" -> Domain: "another-example.co.uk"
    # URL: "http://test-domain.info" -> Domain: "test-domain.info"
