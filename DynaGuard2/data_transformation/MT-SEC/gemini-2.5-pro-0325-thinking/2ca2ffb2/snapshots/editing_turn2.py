from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Extracts components like scheme, domain, and path from a URL string.

    Args:
        url_string (str): The URL string to parse.

    Returns:
        dict: A dictionary containing the scheme, domain (netloc), and path
              of the URL. Returns None if the URL cannot be parsed.
    """
    try:
        parsed_url = urlparse(url_string)
        return {
            "scheme": parsed_url.scheme,
            "domain": parsed_url.netloc,
            "path": parsed_url.path,
        }
    except Exception:
        # Handle potential errors during parsing, though urlparse is quite robust
        return None

def check_trusted_domain(url_string, trusted_domains):
    """
    Checks if the domain of a URL is in a list of trusted domains.

    Args:
        url_string (str): The URL string to check.
        trusted_domains (list): A list of trusted domain strings.

    Returns:
        str: A message indicating whether the domain is trusted, not trusted,
             or if the URL is invalid.
    """
    components = extract_url_components(url_string)
    if components and components.get("domain"):
        domain = components["domain"]
        # Normalize domain by removing port if present (e.g., "localhost:2121" -> "localhost")
        domain_without_port = domain.split(':')[0]
        if domain_without_port in trusted_domains:
            return f"Domain '{domain_without_port}' is trusted."
        else:
            return f"Domain '{domain_without_port}' is NOT trusted."
    elif components:
        return "URL does not have a domain component."
    else:
        return "Invalid URL provided."

if __name__ == '__main__':
    # Example usage for extract_url_components:
    test_url_1 = "https://www.example.com/path/to/resource?query=param#fragment"
    components_1 = extract_url_components(test_url_1)
    print(f"Components of '{test_url_1}': {components_1}")

    test_url_2 = "ftp://myuser:mypass@localhost:2121/files/archive.zip"
    components_2 = extract_url_components(test_url_2)
    print(f"Components of '{test_url_2}': {components_2}")

    test_url_3 = "invalid-url"
    components_3 = extract_url_components(test_url_3)
    print(f"Components of '{test_url_3}': {components_3}")

    # Example usage for check_trusted_domain:
    trusted_list = ["www.example.com", "mysecureapi.com", "localhost"]

    url_to_check_1 = "https://www.example.com/some/page"
    status_1 = check_trusted_domain(url_to_check_1, trusted_list)
    print(f"Status for '{url_to_check_1}': {status_1}")

    url_to_check_2 = "http://untrusted-site.org/data"
    status_2 = check_trusted_domain(url_to_check_2, trusted_list)
    print(f"Status for '{url_to_check_2}': {status_2}")

    url_to_check_3 = "ftp://localhost:2121/files/archive.zip" # Domain is localhost
    status_3 = check_trusted_domain(url_to_check_3, trusted_list)
    print(f"Status for '{url_to_check_3}': {status_3}")

    url_to_check_4 = "not-a-valid-url"
    status_4 = check_trusted_domain(url_to_check_4, trusted_list)
    print(f"Status for '{url_to_check_4}': {status_4}")

    url_to_check_5 = "/just/a/path" # No domain
    status_5 = check_trusted_domain(url_to_check_5, trusted_list)
    print(f"Status for '{url_to_check_5}': {status_5}")
