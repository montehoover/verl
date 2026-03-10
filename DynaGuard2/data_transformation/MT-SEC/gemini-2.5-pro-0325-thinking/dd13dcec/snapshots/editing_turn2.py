from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the scheme, hostname, and path of the URL.
        Returns None for hostname or path if they are not present.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "hostname": parsed_url.hostname,
        "path": parsed_url.path
    }

def is_trusted_domain(url_string, trusted_domains):
    """
    Verifies if the URL's hostname is part of a predefined list of trusted domains.

    Args:
        url_string: The URL string to check.
        trusted_domains: A list or set of trusted domain names.

    Returns:
        True if the URL's hostname is in the trusted_domains list, False otherwise.
        Returns False if the URL has no hostname.
    """
    parts = extract_url_parts(url_string)
    hostname = parts.get("hostname")
    if hostname:
        return hostname in trusted_domains
    return False

if __name__ == '__main__':
    # Example usage for extract_url_parts:
    test_url_1 = "http://www.example.com/path/to/page?query=string#fragment"
    parts_1 = extract_url_parts(test_url_1)
    print(f"URL: {test_url_1}")
    print(f"Scheme: {parts_1['scheme']}")
    print(f"Hostname: {parts_1['hostname']}")
    print(f"Path: {parts_1['path']}")
    print("-" * 20)

    test_url_2 = "https://subdomain.example.co.uk:8080/another/path.html"
    parts_2 = extract_url_parts(test_url_2)
    print(f"URL: {test_url_2}")
    print(f"Scheme: {parts_2['scheme']}")
    print(f"Hostname: {parts_2['hostname']}")
    print(f"Path: {parts_2['path']}")
    print("-" * 20)

    test_url_3 = "ftp://ftp.example.org/resource"
    parts_3 = extract_url_parts(test_url_3)
    print(f"URL: {test_url_3}")
    print(f"Scheme: {parts_3['scheme']}")
    print(f"Hostname: {parts_3['hostname']}")
    print(f"Path: {parts_3['path']}")
    print("-" * 20)
    
    test_url_4 = "example.com/just/path" # scheme missing
    parts_4 = extract_url_parts(test_url_4)
    print(f"URL: {test_url_4}")
    print(f"Scheme: {parts_4['scheme']}") # Will be empty
    print(f"Hostname: {parts_4['hostname']}") # Will be None
    print(f"Path: {parts_4['path']}") # Will be example.com/just/path
    print("-" * 20)

    test_url_5 = "mailto:user@example.com"
    parts_5 = extract_url_parts(test_url_5)
    print(f"URL: {test_url_5}")
    print(f"Scheme: {parts_5['scheme']}")
    print(f"Hostname: {parts_5['hostname']}") # Will be None
    print(f"Path: {parts_5['path']}")
    print("-" * 20)

    # Example usage for is_trusted_domain:
    trusted_domains_list = {"example.com", "example.co.uk", "google.com"}

    url_trusted = "http://example.com/some/page"
    url_untrusted = "http://unknown.org/another/page"
    url_no_hostname = "mailto:test@example.com"
    url_subdomain_trusted = "https://subdomain.example.co.uk/path" # This will be false unless "subdomain.example.co.uk" is in trusted_domains_list

    print(f"Checking if '{url_trusted}' is trusted: {is_trusted_domain(url_trusted, trusted_domains_list)}")
    print(f"Checking if '{url_untrusted}' is trusted: {is_trusted_domain(url_untrusted, trusted_domains_list)}")
    print(f"Checking if '{url_no_hostname}' is trusted: {is_trusted_domain(url_no_hostname, trusted_domains_list)}")
    print(f"Checking if '{url_subdomain_trusted}' is trusted: {is_trusted_domain(url_subdomain_trusted, trusted_domains_list)}")
    print("-" * 20)

    trusted_domains_list_with_subdomain = {"example.com", "subdomain.example.co.uk", "google.com"}
    print(f"Checking if '{url_subdomain_trusted}' is trusted (with subdomain in list): {is_trusted_domain(url_subdomain_trusted, trusted_domains_list_with_subdomain)}")
