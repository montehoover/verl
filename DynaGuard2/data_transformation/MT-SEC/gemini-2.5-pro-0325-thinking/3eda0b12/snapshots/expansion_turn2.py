from urllib.parse import urlparse

# Global variable for permitted domains
PERMITTED_DOMAINS = {
    "example.com",
    "trusted.org",
    "sub.example.com",
}

def is_domain_permitted(url: str) -> bool:
    """
    Checks if the domain of the given URL is in the list of permitted domains.

    Args:
        url: The URL string to check.

    Returns:
        True if the domain is permitted, False otherwise.
    """
    try:
        # First, validate the URL structure.
        # If validate_url raises ValueError, this function will also effectively fail.
        validate_url(url)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Remove port if present for domain matching
        if ":" in domain:
            domain = domain.split(":")[0]
        
        # Check against www. subdomain and the base domain
        if domain in PERMITTED_DOMAINS:
            return True
        if domain.startswith("www.") and domain[4:] in PERMITTED_DOMAINS:
            return True
            
        return False
    except ValueError:
        # If validate_url raises an error, or urlparse fails in an unexpected way
        return False

def validate_url(url: str) -> bool:
    """
    Validates a given URL.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid.

    Raises:
        ValueError: If the URL is invalid.
    """
    try:
        result = urlparse(url)
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"Invalid URL: {url}. Missing scheme or network location.")
    except Exception as e: # Catch any parsing errors from urlparse itself, though less common for basic structure
        raise ValueError(f"Invalid URL: {url}. Parsing error: {e}")

if __name__ == '__main__':
    # Example Usage
    valid_urls = [
        "http://www.example.com",
        "https://example.com/path?query=value#fragment",
        "ftp://user:password@host:port/path",
    ]
    invalid_urls = [
        "www.example.com",
        "example.com",
        "http//example.com",
        "just_a_string",
        "",
        None, # type: ignore
        "http://",
        "http:///path",
    ]

    print("Testing valid URLs:")
    for url_str in valid_urls:
        try:
            if validate_url(url_str):
                print(f"'{url_str}' is valid.")
        except ValueError as e:
            print(f"Error validating '{url_str}': {e}")

    print("\nTesting invalid URLs:")
    for url_str in invalid_urls:
        try:
            validate_url(url_str)
            print(f"'{url_str}' was unexpectedly considered valid.")
        except ValueError as e:
            print(f"Correctly identified '{url_str}' as invalid: {e}")
        except TypeError as e: # Handles None case specifically if type hints are enforced at runtime
             print(f"Correctly identified '{url_str}' as invalid (TypeError): {e}")

    print("\nTesting domain permissions:")
    permitted_domain_urls = [
        "http://example.com/page",
        "https://www.example.com/another/path",
        "http://trusted.org",
        "ftp://sub.example.com/resource",
        "http://example.com:8080/path" # Domain check should ignore port
    ]
    non_permitted_domain_urls = [
        "http://untrusted.com",
        "https://sub.untrusted.org/path",
        "http://anotherexample.com",
        "www.example.com", # Invalid URL, should be caught by validate_url first
    ]

    print("\nTesting permitted domain URLs:")
    for url_str in permitted_domain_urls:
        if is_domain_permitted(url_str):
            print(f"Domain for '{url_str}' is permitted.")
        else:
            print(f"Domain for '{url_str}' is NOT permitted (but was expected to be).")

    print("\nTesting non-permitted domain URLs:")
    for url_str in non_permitted_domain_urls:
        if not is_domain_permitted(url_str):
            print(f"Domain for '{url_str}' is correctly not permitted.")
        else:
            # This case also covers if validate_url itself failed and is_domain_permitted returned False
            if not validate_url(url_str): # Check if it was an invalid URL to begin with
                 print(f"URL '{url_str}' is invalid, thus domain not permitted.")
            else:
                 print(f"Domain for '{url_str}' IS permitted (but was expected NOT to be).")
