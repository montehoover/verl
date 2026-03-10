from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary with URL components like 'scheme', 'domain', 'path', etc.
    """
    parsed_url = urlparse(url_string)
    return {
        'scheme': parsed_url.scheme,
        'domain': parsed_url.netloc,
        'path': parsed_url.path,
        'params': parsed_url.params,
        'query': parsed_url.query,
        'fragment': parsed_url.fragment
    }

ALLOWED_DOMAINS = ["www.example.com", "example.org", "localhost"]

def check_domain_approval(url_string):
    """
    Checks if the domain of a URL is in a predefined list of allowed domains.

    Args:
        url_string: The URL string to check.

    Returns:
        A string indicating whether the domain is 'approved' or 'not approved'.
    """
    components = extract_url_components(url_string)
    domain = components.get('domain')

    if domain and domain in ALLOWED_DOMAINS:
        return "approved"
    else:
        return "not approved"

if __name__ == '__main__':
    # Example usage for extract_url_components:
    test_url_1 = "https://www.example.com/path/to/resource?query=param#fragment"
    components_1 = extract_url_components(test_url_1)
    print(f"Components of {test_url_1}: {components_1}")

    test_url_2 = "http://localhost:8080/api/users?id=123"
    components_2 = extract_url_components(test_url_2)
    print(f"Components of {test_url_2}: {components_2}")

    test_url_3 = "ftp://user:password@ftp.example.com/files/archive.zip"
    components_3 = extract_url_components(test_url_3)
    print(f"Components of {test_url_3}: {components_3}")

    test_url_4 = "/relative/path?key=value" # Relative URL
    components_4 = extract_url_components(test_url_4)
    print(f"Components of {test_url_4}: {components_4}")

    # Example usage for check_domain_approval:
    url_to_check_1 = "https://www.example.com/some/page"
    approval_status_1 = check_domain_approval(url_to_check_1)
    print(f"Domain status for {url_to_check_1}: {approval_status_1}")

    url_to_check_2 = "http://www.unknown.com/test"
    approval_status_2 = check_domain_approval(url_to_check_2)
    print(f"Domain status for {url_to_check_2}: {approval_status_2}")

    url_to_check_3 = "ftp://example.org/files"
    approval_status_3 = check_domain_approval(url_to_check_3)
    print(f"Domain status for {url_to_check_3}: {approval_status_3}")

    url_to_check_4 = "http://localhost:8080/api"
    approval_status_4 = check_domain_approval(url_to_check_4)
    print(f"Domain status for {url_to_check_4}: {approval_status_4}")
