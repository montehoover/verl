from urllib.parse import urlparse, urljoin

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

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
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    try:
        result = urlparse(url)
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"Invalid URL: '{url}'. Missing scheme or network location.")
    except Exception as e: # Catch any parsing errors from urlparse itself, though it's usually robust.
        # More specific error handling for urlparse issues could be added if needed.
        # For now, re-raising as ValueError or letting specific ValueErrors from above propagate.
        if isinstance(e, ValueError): # if it's already our ValueError
            raise
        raise ValueError(f"Invalid URL format: '{url}'. Error: {e}")

def construct_resource_path(res_id: str, res_type: str) -> str:
    """
    Constructs a resource path based on resource ID and type.

    Args:
        res_id: The identifier of the resource.
        res_type: The type of the resource.

    Returns:
        A string representing the resource path, e.g., "/<res_type>/<res_id>".
    
    Raises:
        TypeError: If res_id or res_type are not strings.
        ValueError: If res_id or res_type are empty strings.
    """
    if not isinstance(res_id, str) or not isinstance(res_type, str):
        raise TypeError("Resource ID and type must be strings.")
    if not res_id:
        raise ValueError("Resource ID cannot be empty.")
    if not res_type:
        raise ValueError("Resource type cannot be empty.")
    
    # Basic path construction.
    # For URL paths, direct string formatting is common.
    # Ensure no leading/trailing slashes in res_type and res_id if they might be user-provided
    # and could break the path structure, e.g. res_type.strip('/')
    return f"/{res_type.strip('/')}/{res_id.strip('/')}"

def create_resource_url(base_cdn_url: str, res_id: str, res_type: str) -> str:
    """
    Creates a complete URL for a resource on a CDN.

    Args:
        base_cdn_url: The base URL of the CDN.
        res_id: The identifier of the resource.
        res_type: The type of the resource.

    Returns:
        A string representing the complete URL for the resource.

    Raises:
        ValueError: If the base_cdn_url is invalid, its domain is not allowed,
                    or if res_id or res_type are invalid for path construction.
        TypeError: If arguments are of incorrect types.
    """
    # Validate the base_cdn_url itself
    validate_url(base_cdn_url) # This will raise ValueError if base_cdn_url is malformed

    parsed_url = urlparse(base_cdn_url)
    if parsed_url.netloc not in ALLOWED_CDN_DOMAINS:
        raise ValueError(f"Domain '{parsed_url.netloc}' is not an allowed CDN domain.")

    # Construct the resource-specific path part
    # construct_resource_path will raise TypeError/ValueError for invalid res_id/res_type
    resource_path = construct_resource_path(res_id, res_type)

    # Ensure base_cdn_url ends with a slash for proper joining if it's just a domain
    # urljoin handles this well, but being explicit can be clearer.
    # If base_cdn_url is "http://cdn.example.com" and resource_path is "/product/item123",
    # urljoin("http://cdn.example.com", "/product/item123") -> "http://cdn.example.com/product/item123"
    # If base_cdn_url is "http://cdn.example.com/" and resource_path is "/product/item123",
    # urljoin("http://cdn.example.com/", "/product/item123") -> "http://cdn.example.com/product/item123"
    # If resource_path starts with '/', urljoin treats it as an absolute path relative to the domain.
    
    # urljoin is generally robust for this.
    # If resource_path might not start with a '/', we might need to adjust.
    # Our construct_resource_path ensures it starts with '/'.
    complete_url = urljoin(base_cdn_url, resource_path)
    
    return complete_url

if __name__ == '__main__':
    # Example Usage
    valid_urls = [
        "http://www.example.com",
        "https://example.com/path?query=value#fragment",
        "ftp://user:password@host.com:21/path",
    ]

    invalid_urls = [
        "www.example.com",  # Missing scheme
        "http://",  # Missing netloc
        "just_a_string",
        None, # type error
        123, # type error
        "http:///path", # malformed netloc
        "://example.com" # missing scheme
    ]

    print("Testing valid URLs:")
    for u in valid_urls:
        try:
            validate_url(u)
            print(f"'{u}' is valid.")
        except ValueError as e:
            print(f"'{u}' is invalid (unexpected): {e}")

    print("\nTesting invalid URLs:")
    for u in invalid_urls:
        try:
            validate_url(u)
            print(f"'{u}' is valid (unexpected).")
        except ValueError as e:
            print(f"'{u}' is invalid: {e}")
        except TypeError as e: # For None, 123
             print(f"'{u}' caused TypeError: {e}")

    print("\nTesting construct_resource_path:")
    test_cases_construct_path = [
        ("item123", "product", "/product/item123"),
        ("user456", "user", "/user/user456"),
        ("doc789", "document", "/document/doc789"),
        ("id_with_slash/", "/type_with_slash", "/type_with_slash/id_with_slash"), # Test stripping
    ]

    for res_id, res_type, expected_path in test_cases_construct_path:
        try:
            path = construct_resource_path(res_id, res_type)
            if path == expected_path:
                print(f"construct_resource_path('{res_id}', '{res_type}') -> '{path}' (Correct)")
            else:
                print(f"construct_resource_path('{res_id}', '{res_type}') -> '{path}' (Incorrect, expected '{expected_path}')")
        except (ValueError, TypeError) as e:
            print(f"construct_resource_path('{res_id}', '{res_type}') -> Error: {e} (Incorrect, expected '{expected_path}')")

    # Test error cases for construct_resource_path
    print("\nTesting construct_resource_path error cases:")
    error_test_cases_construct = [
        (123, "product", TypeError),
        ("item123", None, TypeError),
        ("", "product", ValueError),
        ("item123", "", ValueError),
    ]
    for res_id, res_type, expected_error in error_test_cases_construct:
        try:
            path = construct_resource_path(res_id, res_type)
            print(f"construct_resource_path('{res_id}', '{res_type}') -> '{path}' (Error not raised as expected: {expected_error.__name__})")
        except Exception as e:
            if isinstance(e, expected_error):
                print(f"construct_resource_path('{res_id}', '{res_type}') -> Correctly raised error: {e}")
            else:
                print(f"construct_resource_path('{res_id}', '{res_type}') -> Incorrect error type raised: {type(e).__name__}, expected {expected_error.__name__}. Error: {e}")

    print("\nTesting create_resource_url:")
    test_cases_create_url = [
        ("http://cdn.example.com", "item123", "product", "http://cdn.example.com/product/item123"),
        ("https://assets.example.net/v1/", "user456", "user", "https://assets.example.net/v1/user/user456"),
        ("http://static.example.org", "doc789", "document", "http://static.example.org/document/doc789"),
    ]

    for base_url, res_id, res_type, expected_url in test_cases_create_url:
        try:
            url = create_resource_url(base_url, res_id, res_type)
            if url == expected_url:
                print(f"create_resource_url('{base_url}', '{res_id}', '{res_type}') -> '{url}' (Correct)")
            else:
                print(f"create_resource_url('{base_url}', '{res_id}', '{res_type}') -> '{url}' (Incorrect, expected '{expected_url}')")
        except (ValueError, TypeError) as e:
            print(f"create_resource_url('{base_url}', '{res_id}', '{res_type}') -> Error: {e} (Incorrect, expected '{expected_url}')")

    print("\nTesting create_resource_url error cases:")
    error_test_cases_create_url = [
        ("http://invalid-domain.com", "item123", "product", ValueError), # Invalid domain
        ("cdn.example.com", "item123", "product", ValueError), # Invalid base_cdn_url (missing scheme)
        ("http://cdn.example.com", "", "product", ValueError), # Invalid res_id
        ("http://cdn.example.com", "item123", None, TypeError), # Invalid res_type
        (None, "item123", "product", ValueError), # Invalid base_cdn_url (None, validate_url handles this)
    ]

    for base_url, res_id, res_type, expected_error in error_test_cases_create_url:
        try:
            url = create_resource_url(base_url, res_id, res_type)
            print(f"create_resource_url('{base_url}', '{res_id}', '{res_type}') -> '{url}' (Error not raised as expected: {expected_error.__name__})")
        except Exception as e:
            if isinstance(e, expected_error):
                print(f"create_resource_url('{base_url}', '{res_id}', '{res_type}') -> Correctly raised error: {e}")
            else:
                print(f"create_resource_url('{base_url}', '{res_id}', '{res_type}') -> Incorrect error type raised: {type(e).__name__}, expected {expected_error.__name__}. Error: {e}")
