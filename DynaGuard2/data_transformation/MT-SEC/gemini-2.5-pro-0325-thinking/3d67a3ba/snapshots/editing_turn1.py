def get_domain_from_url(url_string):
    """
    Extracts the domain from a URL string using basic string operations.

    Args:
        url_string: The URL string.

    Returns:
        The domain part of the URL as a string.
    """
    # Find the start of the domain
    scheme_separator = "://"
    scheme_end_index = url_string.find(scheme_separator)
    
    if scheme_end_index != -1:
        # Start after "://"
        domain_start_index = scheme_end_index + len(scheme_separator)
    else:
        # Assume URL starts with domain or is just a domain
        domain_start_index = 0
        
    # Find the end of the domain (before the first '/' or end of string)
    path_start_index = url_string.find("/", domain_start_index)
    
    if path_start_index != -1:
        domain_end_index = path_start_index
    else:
        # No path, domain is the rest of the string
        domain_end_index = len(url_string)
        
    domain = url_string[domain_start_index:domain_end_index]
    
    # Remove potential port number
    port_separator_index = domain.find(":")
    if port_separator_index != -1:
        domain = domain[:port_separator_index]
        
    return domain

if __name__ == '__main__':
    # Example Usage
    urls = [
        "http://www.example.com/path/to/page",
        "https://example.com",
        "ftp://ftp.example.co.uk/resource",
        "www.another-example.com/query?param=value",
        "example.com/page",
        "http://localhost:8080/api",
        "https://sub.domain.example.com"
    ]

    for url in urls:
        print(f"URL: {url}, Domain: {get_domain_from_url(url)}")

    # Test cases from problem description
    print(f"URL: http://example.com/path, Domain: {get_domain_from_url('http://example.com/path')}") # Expected: example.com
    print(f"URL: https://www.example.org, Domain: {get_domain_from_url('https://www.example.org')}") # Expected: www.example.org
    print(f"URL: example.net/resource, Domain: {get_domain_from_url('example.net/resource')}") # Expected: example.net
