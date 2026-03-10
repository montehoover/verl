from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the URL components (scheme, netloc, path, params, query, fragment).
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,  # Domain name
        "path": parsed_url.path,
        "params": parsed_url.params,
        "query": parsed_url.query,
        "fragment": parsed_url.fragment,
    }

def check_domain_approval(url_string, allowed_domains):
    """
    Checks if the domain of a URL is in a list of allowed domains.

    Args:
        url_string: The URL string to check.
        allowed_domains: A list or set of allowed domain names.

    Returns:
        A dictionary containing:
            - "approved": A boolean indicating if the domain is approved.
            - "message": A string message ("approved" or "not approved").
            - "domain": The extracted domain.
    """
    components = extract_url_components(url_string)
    domain = components.get("netloc")

    if domain and domain in allowed_domains:
        return {"approved": True, "message": "approved", "domain": domain}
    else:
        return {"approved": False, "message": "not approved", "domain": domain}

if __name__ == '__main__':
    # Example usage for extract_url_components:
    test_url = "https://www.example.com/path/to/resource?param1=value1&param2=value2#section1"
    components = extract_url_components(test_url)
    print(f"URL: {test_url}")
    print(f"Components: {components}")

    test_url_2 = "http://localhost:8080/api/users?id=123"
    components_2 = extract_url_components(test_url_2)
    print(f"\nURL: {test_url_2}")
    print(f"Components: {components_2}")

    test_url_3 = "ftp://user:password@host.com/data.txt"
    components_3 = extract_url_components(test_url_3)
    print(f"\nURL: {test_url_3}")
    print(f"Components: {components_3}")

    # Example usage for check_domain_approval:
    allowed_domains_list = {"www.example.com", "api.example.org", "localhost"}

    url_to_check_1 = "https://www.example.com/some/page"
    approval_status_1 = check_domain_approval(url_to_check_1, allowed_domains_list)
    print(f"\nURL: {url_to_check_1}")
    print(f"Domain: {approval_status_1['domain']}, Status: {approval_status_1['message']} (Approved: {approval_status_1['approved']})")

    url_to_check_2 = "http://unknown-domain.net/test"
    approval_status_2 = check_domain_approval(url_to_check_2, allowed_domains_list)
    print(f"\nURL: {url_to_check_2}")
    print(f"Domain: {approval_status_2['domain']}, Status: {approval_status_2['message']} (Approved: {approval_status_2['approved']})")

    url_to_check_3 = "ftp://user:password@host.com/data.txt" # host.com is not in allowed_domains_list
    approval_status_3 = check_domain_approval(url_to_check_3, allowed_domains_list)
    print(f"\nURL: {url_to_check_3}")
    print(f"Domain: {approval_status_3['domain']}, Status: {approval_status_3['message']} (Approved: {approval_status_3['approved']})")

    url_to_check_4 = "http://localhost:8080/api/data" # localhost is in allowed_domains_list
    approval_status_4 = check_domain_approval(url_to_check_4, allowed_domains_list)
    print(f"\nURL: {url_to_check_4}")
    print(f"Domain: {approval_status_4['domain']}, Status: {approval_status_4['message']} (Approved: {approval_status_4['approved']})")
