from urllib.parse import urlparse, urlencode, urlunparse, parse_qs

APPROVED_DOMAINS = ['example.com', 'myapp.com', 'trusted.com']

def construct_and_validate_url(base_url: str, params: dict) -> str:
    """
    Constructs a URL with query parameters and validates its base domain.

    Args:
        base_url: The base URL string.
        params: A dictionary of query parameters to add or override.

    Returns:
        The full URL string with (updated) query parameters if the base domain is approved.

    Raises:
        ValueError: If the base_url is invalid, its domain is not in APPROVED_DOMAINS,
                    or if URL construction fails.
    """
    try:
        parsed_base = urlparse(base_url)
        domain = parsed_base.netloc

        if not domain:
            raise ValueError(f"Invalid base_url: '{base_url}' does not contain a valid domain.")

        effective_domain = domain
        if effective_domain.startswith('www.'):
            effective_domain = effective_domain[4:]

        if effective_domain not in APPROVED_DOMAINS:
            raise ValueError(f"Domain '{domain}' (from base_url '{base_url}') is not an approved domain.")

        # Merge existing query parameters from base_url with new params
        # New params will override existing ones with the same key
        query_dict = parse_qs(parsed_base.query, keep_blank_values=True)
        for key, value in params.items():
            query_dict[key] = [str(value)]  # Store as list of strings for urlencode

        new_query_string = urlencode(query_dict, doseq=True)
        
        full_url = urlunparse(parsed_base._replace(query=new_query_string))
        return full_url

    except ValueError: # Re-raise ValueError from domain validation or invalid base_url
        raise
    except Exception as e: # Catch other potential errors (e.g., from urlparse, urlencode)
        raise ValueError(f"Error constructing or validating URL for '{base_url}': {e}")

if __name__ == '__main__':
    test_cases = [
        ("http://example.com/path", {'p1': 'v1', 'p2': 'v2'}, "Valid: Basic case"),
        ("https://www.myapp.com/api", {'token': 'xyz', 'id': 123}, "Valid: www domain"),
        ("ftp://trusted.com/files", {'user': 'anonymous'}, "Valid: ftp scheme"),
        ("http://untrusted.com/data", {'key': 'value'}, "Invalid: Domain not approved"),
        ("http://sub.example.com/path", {'a': 'b'}, "Invalid: Subdomain not explicitly approved"),
        ("example.com/path", {'q': '1'}, "Invalid: Malformed base_url (no scheme/netloc)"),
        ("http://www.example.com", {}, "Valid: Empty params, www domain"),
        ("https://example.com/page?existing=true", {'new_param': 'added', 'existing': 'overwritten'}, "Valid: Merge/overwrite params"),
        ("http://myapp.com", {'id': 42, 'name': 'test item'}, "Valid: Base URL no path, add params"),
        ("not_a_url_at_all", {'p': '1'}, "Invalid: Malformed base_url"),
        ("http://trusted.com/action?name=alpha&val=1", {'val': '2', 'opt': 'beta'}, "Valid: Complex param merge")
    ]

    for base_url, params, description in test_cases:
        print(f"Test Case: {description}")
        print(f"Base URL: '{base_url}', Params: {params}")
        try:
            valid_url = construct_and_validate_url(base_url, params)
            print(f"Constructed URL: {valid_url}")
        except ValueError as e:
            print(f"Error: {e}")
        print("-" * 30)

    # Example of how parameters are merged/overwritten
    print("Test Case: Parameter merging details")
    base = "http://example.com/search?type=product&color=blue"
    p = {'category': 'electronics', 'type': 'gadget'} # 'type' will be overwritten, 'color' preserved, 'category' added
    print(f"Base URL: '{base}', Params: {p}")
    try:
        # Expected: http://example.com/search?type=gadget&color=blue&category=electronics (order of params may vary)
        valid_url = construct_and_validate_url(base, p)
        print(f"Constructed URL: {valid_url}")
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 30)
