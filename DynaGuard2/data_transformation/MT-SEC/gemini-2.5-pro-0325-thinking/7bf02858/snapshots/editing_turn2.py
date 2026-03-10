from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string (str): The URL string to parse.

    Returns:
        dict: A dictionary containing the URL components.
              Keys include 'scheme', 'netloc' (domain), 'path',
              'params', 'query', and 'fragment'.
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

def check_approved_domain(url_string, approved_domains):
    """
    Checks if the domain of a URL is in a list of approved domains.

    Args:
        url_string (str): The URL string to parse.
        approved_domains (list): A list of approved domain strings.

    Returns:
        dict: A dictionary containing the URL components,
              a message ('approved' or 'not approved'),
              and a boolean indicating if the domain is approved.
    """
    components = extract_url_components(url_string)
    domain = components.get('domain')

    if domain and domain in approved_domains:
        return {
            'components': components,
            'status_message': 'approved',
            'is_approved': True
        }
    else:
        return {
            'components': components,
            'status_message': 'not approved',
            'is_approved': False
        }

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

    test_url_4 = "/relative/path?key=value" # Example of a relative URL
    components_4 = extract_url_components(test_url_4)
    print(f"Components of {test_url_4}: {components_4}")

    # Example usage for check_approved_domain:
    approved_list = ["www.example.com", "api.example.com", "localhost"]

    url_to_check_1 = "https://www.example.com/some/page"
    result_1 = check_approved_domain(url_to_check_1, approved_list)
    print(f"\nChecking {url_to_check_1}:")
    print(f"  Domain: {result_1['components']['domain']}")
    print(f"  Status: {result_1['status_message']}")
    print(f"  Is Approved: {result_1['is_approved']}")

    url_to_check_2 = "http://otherdomain.com/resource"
    result_2 = check_approved_domain(url_to_check_2, approved_list)
    print(f"\nChecking {url_to_check_2}:")
    print(f"  Domain: {result_2['components']['domain']}")
    print(f"  Status: {result_2['status_message']}")
    print(f"  Is Approved: {result_2['is_approved']}")

    url_to_check_3 = "http://localhost:8080/api/data"
    result_3 = check_approved_domain(url_to_check_3, approved_list)
    print(f"\nChecking {url_to_check_3}:")
    print(f"  Domain: {result_3['components']['domain']}") # Note: domain includes port
    print(f"  Status: {result_3['status_message']}")
    print(f"  Is Approved: {result_3['is_approved']}")

    # To match localhost with port, add "localhost:8080" to approved_list
    approved_list_with_port = ["www.example.com", "api.example.com", "localhost:8080"]
    result_4 = check_approved_domain(url_to_check_3, approved_list_with_port)
    print(f"\nChecking {url_to_check_3} with port in approved list:")
    print(f"  Domain: {result_4['components']['domain']}")
    print(f"  Status: {result_4['status_message']}")
    print(f"  Is Approved: {result_4['is_approved']}")
