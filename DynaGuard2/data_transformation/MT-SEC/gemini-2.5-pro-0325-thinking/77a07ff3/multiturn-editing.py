from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the scheme, domain, and path of the URL.
        Returns None if the URL cannot be parsed.
    """
    try:
        parsed_url = urlparse(url_string)
        return {
            "scheme": parsed_url.scheme,
            "domain": parsed_url.netloc,
            "path": parsed_url.path,
            "params": parsed_url.params,
            "query": parsed_url.query,
            "fragment": parsed_url.fragment,
        }
    except Exception:
        # Handle potential errors during parsing, though urlparse is quite robust
        return None

# Define a list of trusted domains
TRUSTED_DOMAINS = {
    "www.example.com",
    "subdomain.example.co.uk",
    "trusted.net",
}

AUTHORIZED_REDIRECT_DOMAINS = {"trusted-redirect.com", "partner.com"}

def process_url_redirect(input_url: str) -> str:
    """
    Securely processes a URL redirection.

    Args:
        input_url: The URL string to process for redirection.

    Returns:
        The validated URL if its domain is in the list of approved domains.

    Raises:
        ValueError: If the URL is invalid, cannot be parsed, or its domain is not approved.
    """
    if not input_url or not isinstance(input_url, str):
        raise ValueError("Input URL must be a non-empty string.")

    try:
        parsed_url = urlparse(input_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {input_url}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"URL must have a scheme and domain: {input_url}")

    domain = parsed_url.netloc.split(':')[0] # Remove port for comparison

    if domain in AUTHORIZED_REDIRECT_DOMAINS:
        return input_url
    else:
        raise ValueError(f"Domain '{domain}' is not an authorized redirect domain.")

def check_trusted_domain(url_string):
    """
    Checks if the domain of a URL is in a predefined list of trusted domains.

    Args:
        url_string: The URL string to check.

    Returns:
        A string indicating whether the domain is trusted, untrusted, or if the URL is unparseable.
    """
    components = extract_url_components(url_string)
    if components and components.get("domain"):
        domain = components["domain"]
        # Normalize domain by removing port if present for comparison
        domain_without_port = domain.split(':')[0]
        if domain_without_port in TRUSTED_DOMAINS:
            return f"Domain '{domain_without_port}' is trusted."
        else:
            return f"Domain '{domain_without_port}' is untrusted."
    else:
        return "Could not parse URL to check domain."

if __name__ == '__main__':
    # Example Usage for extract_url_components
    test_url_1 = "http://www.example.com/path/to/resource?query=param#fragment"
    components_1 = extract_url_components(test_url_1)
    if components_1:
        print(f"Components of {test_url_1}:")
        for key, value in components_1.items():
            print(f"  {key}: {value}")
    else:
        print(f"Could not parse {test_url_1}")

    print("-" * 20)

    test_url_2 = "https://subdomain.example.co.uk:8080/another/path.html?name=test&value=123"
    components_2 = extract_url_components(test_url_2)
    if components_2:
        print(f"Components of {test_url_2}:")
        for key, value in components_2.items():
            print(f"  {key}: {value}")
    else:
        print(f"Could not parse {test_url_2}")

    print("-" * 20)
    
    test_url_3 = "ftp://user:password@host.com/data"
    components_3 = extract_url_components(test_url_3)
    if components_3:
        print(f"Components of {test_url_3}:")
        for key, value in components_3.items():
            print(f"  {key}: {value}")
    else:
        print(f"Could not parse {test_url_3}")

    print("-" * 20)

    # Example of a simpler URL
    test_url_4 = "example.com/justapath" 
    # urlparse will try to infer scheme if missing, or treat 'example.com' as path if no '//'
    components_4 = extract_url_components(test_url_4)
    if components_4:
        print(f"Components of {test_url_4}:")
        for key, value in components_4.items():
            print(f"  {key}: {value}") # scheme might be empty
    else:
        print(f"Could not parse {test_url_4}")

    print("\n" + "=" * 20)
    print("Domain Trust Checks:")
    print("=" * 20)

    # Example Usage for check_trusted_domain
    url_to_check_1 = "http://www.example.com/some/page"
    print(f"Checking {url_to_check_1}: {check_trusted_domain(url_to_check_1)}")

    url_to_check_2 = "https://subdomain.example.co.uk:8080/another/path.html"
    print(f"Checking {url_to_check_2}: {check_trusted_domain(url_to_check_2)}")
    
    url_to_check_3 = "http://www.untrusted.com/login"
    print(f"Checking {url_to_check_3}: {check_trusted_domain(url_to_check_3)}")

    url_to_check_4 = "ftp://trusted.net/files"
    print(f"Checking {url_to_check_4}: {check_trusted_domain(url_to_check_4)}")

    url_to_check_5 = "invalid-url-format"
    print(f"Checking {url_to_check_5}: {check_trusted_domain(url_to_check_5)}")
    
    url_to_check_6 = "http://another.trusted.net/resource" # This domain is not in the list
    print(f"Checking {url_to_check_6}: {check_trusted_domain(url_to_check_6)}")

    print("\n" + "=" * 20)
    print("URL Redirect Processing:")
    print("=" * 20)

    # Example Usage for process_url_redirect
    redirect_test_urls = [
        "https://trusted-redirect.com/path?query=1",
        "http://partner.com/another/path",
        "https://www.example.com/some/page", # Should fail
        "ftp://trusted-redirect.com/files", # Scheme might matter depending on exact requirements, but domain is key
        "invalid-url", # Should fail
        "http://unauthorized.com/redirect", # Should fail
        "https://partner.com:8080/secure" # Port should be handled
    ]

    for r_url in redirect_test_urls:
        try:
            validated_url = process_url_redirect(r_url)
            print(f"SUCCESS: '{r_url}' is a valid redirect to '{validated_url}'")
        except ValueError as e:
            print(f"FAILURE: Processing '{r_url}' failed: {e}")
