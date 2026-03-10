from urllib.parse import urlparse, urlencode, urljoin, parse_qsl, urlunparse

# Define a list of trusted domains
TRUSTED_DOMAINS = [
    "example.com",
    "trusted.org",
    "another.trusted.net",
]

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def validate_url(url: str) -> bool:
    """
    Validates a URL and checks if it belongs to a trusted domain.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid and belongs to a trusted domain, False otherwise.

    Raises:
        ValueError: If the URL is malformed or has an invalid scheme.
    """
    if not isinstance(url, str):
        raise TypeError("URL must be a string.")

    try:
        parsed_url = urlparse(url)
    except Exception as e: # Catch any parsing errors, though urlparse is quite robust
        raise ValueError(f"URL parsing failed: {e}")

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("URL is malformed. It must include a scheme (e.g., http, https) and a domain.")

    if parsed_url.scheme not in ["http", "https"]:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Only 'http' and 'https' are allowed.")

    # Extract the domain (netloc)
    domain = parsed_url.netloc
    # Remove port if present (e.g., example.com:8080 -> example.com)
    if ":" in domain:
        domain = domain.split(":")[0]

    if domain in TRUSTED_DOMAINS:
        return True
    
    # Check for subdomains of trusted domains if needed, e.g. api.example.com
    # For simplicity, this example only checks exact domain matches.
    # If subdomains are allowed, the logic would be:
    # for trusted_domain in TRUSTED_DOMAINS:
    #     if domain == trusted_domain or domain.endswith("." + trusted_domain):
    #         return True
            
    return False

def construct_query_parameters(params: dict) -> str:
    """
    Constructs a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of parameters (key-value pairs).

    Returns:
        A URL-encoded query string.
    """
    if not isinstance(params, dict):
        raise TypeError("Parameters must be a dictionary.")
    
    # Ensure all keys and values are strings, or handle other types as needed.
    # For simplicity, this example assumes string keys and values,
    # or types that urlencode can handle directly.
    return urlencode(params)

def construct_oauth_callback_uri(root_url: str, callback_uri: str, session_state: str) -> str:
    """
    Constructs a complete OAuth callback URL, validating the callback domain.

    Args:
        root_url: The base URL of the OAuth provider (e.g., "https://auth.myapp.com").
        callback_uri: The path for the callback (e.g., "/oauth/callback").
        session_state: The state parameter to include in the callback URL.

    Returns:
        The complete OAuth callback URL string.

    Raises:
        ValueError: If the callback_uri's domain is not in ALLOWED_CALLBACK_DOMAINS
                    or if URLs are malformed.
        TypeError: If input arguments are not strings.
    """
    if not all(isinstance(arg, str) for arg in [root_url, callback_uri, session_state]):
        raise TypeError("All arguments (root_url, callback_uri, session_state) must be strings.")

    # Ensure callback_uri starts with a slash if it's meant to be a path
    if not callback_uri.startswith("/"):
        callback_uri = "/" + callback_uri
    
    # Construct a temporary full URL for the callback_uri to parse its domain
    # We assume root_url provides the scheme and domain for the callback.
    # Or, if callback_uri is a full URL, we use that.
    
    # Let's parse the root_url to get its scheme and netloc for validation
    parsed_root_url = urlparse(root_url)
    if not parsed_root_url.scheme or not parsed_root_url.netloc:
        raise ValueError("root_url is malformed. It must include a scheme and domain.")

    callback_domain = parsed_root_url.netloc
    if ":" in callback_domain: # remove port if present
        callback_domain = callback_domain.split(":")[0]

    if callback_domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(
            f"Callback domain '{callback_domain}' is not allowed. "
            f"Allowed domains are: {ALLOWED_CALLBACK_DOMAINS}"
        )

    # Construct the final URL
    # urljoin handles joining root_url and callback_uri correctly
    base_callback_url = urljoin(root_url, callback_uri)
    
    # Add the session_state as a query parameter
    # We need to parse the base_callback_url to safely add/update query parameters
    url_parts = list(urlparse(base_callback_url))
    query = dict(parse_qsl(url_parts[4]))
    query['state'] = session_state
    url_parts[4] = urlencode(query)
    
    return urlunparse(url_parts)

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "https://example.com/path/to/resource",
        "http://trusted.org",
        "https://sub.example.com/another?query=param", # This will be false unless subdomains are handled
        "ftp://untrusted.com",
        "https://another.trusted.net/secure",
        "https://malicious-site.com",
        "http://example.com:8080/path",
        "justadomain.com/path", # Invalid, no scheme
        "http:///path-only", # Invalid, no domain
        "https://", # Invalid
        "", # Invalid
        None, # Will raise TypeError
        123, # Will raise TypeError
    ]

    for test_url in test_urls:
        try:
            is_valid = validate_url(test_url)
            print(f"URL: '{test_url}', Valid and Trusted: {is_valid}")
        except (ValueError, TypeError) as e:
            print(f"URL: '{test_url}', Error: {e}")

    print("\nTesting with a trusted subdomain (requires subdomain logic in validate_url to pass):")
    # To make this pass, you would need to adjust TRUSTED_DOMAINS or the logic in validate_url
    # For example, add "sub.trusted.org" to TRUSTED_DOMAINS or implement subdomain checking.
    try:
        is_valid = validate_url("https://sub.trusted.org/api")
        print(f"URL: 'https://sub.trusted.org/api', Valid and Trusted: {is_valid}")
    except (ValueError, TypeError) as e:
        print(f"URL: 'https://sub.trusted.org/api', Error: {e}")

    print("\nTesting construct_query_parameters:")
    oauth_params = {
        "client_id": "your_client_id",
        "response_type": "code",
        "scope": "openid profile email",
        "redirect_uri": "https://example.com/callback",
        "state": "aRandomGeneratedStateToken123!@#$",
        "nonce": "anotherRandomNonceValue789*&^"
    }
    try:
        query_string = construct_query_parameters(oauth_params)
        print(f"OAuth Parameters: {oauth_params}")
        print(f"Encoded Query String: {query_string}")

        # Example of appending to a base URL
        base_auth_url = "https://trusted.org/oauth/authorize"
        full_auth_url = f"{base_auth_url}?{query_string}"
        print(f"Full Authorization URL: {full_auth_url}")

        # Test with potentially problematic characters
        complex_params = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "message": "Hello & Welcome!",
            "search": "python programming"
        }
        query_string_complex = construct_query_parameters(complex_params)
        print(f"\nComplex Parameters: {complex_params}")
        print(f"Encoded Query String (Complex): {query_string_complex}")

    except TypeError as e:
        print(f"Error constructing query parameters: {e}")
    
    try:
        print("\nTesting construct_query_parameters with invalid input:")
        construct_query_parameters("not a dict")
    except TypeError as e:
        print(f"Error (expected): {e}")

    print("\nTesting construct_oauth_callback_uri:")
    test_oauth_callbacks = [
        ("https://auth.myapp.com", "/oauth/callback", "state123", True),
        ("https://login.myapp.org", "myhandler/callback", "state456", True),
        ("https://oauth.myapp.net:8443", "/login/cb", "state789", True), # Domain check ignores port
        ("https://untrusted.com", "/oauth/callback", "stateABC", False),
        ("http://auth.myapp.com", "/callback", "stateXYZ", True), # Scheme doesn't matter for domain check here
        ("auth.myapp.com", "/callback", "stateDEF", False), # Invalid root_url (no scheme)
    ]

    for root, cb_uri, state, should_succeed in test_oauth_callbacks:
        try:
            full_cb_url = construct_oauth_callback_uri(root, cb_uri, state)
            if not should_succeed:
                print(f"Test FAILED (should have raised error): {root}, {cb_uri} -> {full_cb_url}")
            else:
                print(f"Generated OAuth Callback URL: {full_cb_url}")
                # Validate the generated URL itself (optional, but good practice)
                is_valid_generated = validate_url(full_cb_url) # This uses TRUSTED_DOMAINS, not ALLOWED_CALLBACK_DOMAINS
                print(f"Is generated URL generally valid (using TRUSTED_DOMAINS for its domain '{urlparse(full_cb_url).netloc}')? {is_valid_generated}")

        except (ValueError, TypeError) as e:
            if should_succeed:
                print(f"Test FAILED ({root}, {cb_uri}): {e}")
            else:
                print(f"Error (expected for {root}, {cb_uri}): {e}")
    
    # Test with non-string inputs
    try:
        print("\nTesting construct_oauth_callback_uri with invalid input types:")
        construct_oauth_callback_uri("https://auth.myapp.com", "/callback", 123)
    except TypeError as e:
        print(f"Error (expected): {e}")
