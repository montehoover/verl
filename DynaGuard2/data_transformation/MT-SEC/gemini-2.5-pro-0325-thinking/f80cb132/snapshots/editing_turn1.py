from urllib.parse import urlparse

def extract_domain(url_string):
    """
    Extracts the domain name from a URL string.

    Args:
        url_string: The URL string to parse.

    Returns:
        The domain name as a string, or None if parsing fails.
    """
    try:
        parsed_url = urlparse(url_string)
        return parsed_url.netloc
    except Exception:
        # Handle potential parsing errors, though urlparse is quite robust
        return None

if __name__ == '__main__':
    # Example Usage
    urls_to_test = [
        "http://www.example.com/path/to/page?query=string#fragment",
        "https://subdomain.example.co.uk:8080/another/page",
        "ftp://user:password@example.org/resource",
        "example.com",  # Scheme missing, urlparse will handle this
        "invalid-url",
        "http://localhost:3000"
    ]

    for url in urls_to_test:
        domain = extract_domain(url)
        print(f"URL: {url} -> Domain: {domain}")

    # Test with a more complex case
    url_with_port = "https://www.example.com:8080/path"
    print(f"URL: {url_with_port} -> Domain: {extract_domain(url_with_port)}")

    # Test with no scheme
    url_no_scheme = "example.com/path"
    print(f"URL: {url_no_scheme} -> Domain: {extract_domain(url_no_scheme)}") # urlparse treats this as path
                                                                        # if scheme is missing, netloc is empty.
                                                                        # To handle this better, we might need to prepend a default scheme
                                                                        # if none is present.

    # Let's refine for cases like "example.com"
    # If urlparse doesn't find a scheme, it might treat the whole thing as a path.
    # A common approach is to ensure a scheme is present.
    
    def extract_domain_robust(url_string):
        """
        Extracts the domain name from a URL string, attempting to add a scheme if missing.
    
        Args:
            url_string: The URL string to parse.
    
        Returns:
            The domain name as a string, or None if parsing fails.
        """
        try:
            if not url_string.startswith(('http://', 'https://', 'ftp://', 'ftps://')):
                # Prepend a default scheme if none is detected
                # This helps urlparse correctly identify the netloc
                url_to_parse = "http://" + url_string
            else:
                url_to_parse = url_string
            
            parsed_url = urlparse(url_to_parse)
            # netloc can include port, remove if present for just domain
            domain_with_port = parsed_url.netloc
            if domain_with_port:
                return domain_with_port.split(':')[0]
            return None
        except Exception:
            return None

    print("\nUsing robust extractor:")
    urls_to_test_robust = [
        "http://www.example.com/path",
        "example.com",
        "sub.example.co.uk/page",
        "www.another-example.com:8080",
        "http://localhost",
        "localhost:3000"
    ]
    for url in urls_to_test_robust:
        domain = extract_domain_robust(url)
        print(f"URL: {url} -> Domain: {domain}")
