from urllib.parse import urlparse

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
