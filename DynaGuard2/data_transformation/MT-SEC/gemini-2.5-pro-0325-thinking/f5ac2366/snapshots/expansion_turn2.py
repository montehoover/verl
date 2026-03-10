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
    try:
        result = urlparse(url)
        # A valid URL must have a scheme (e.g., http, https) and a netloc (e.g., www.example.com)
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"Invalid URL: {url}. Scheme and network location are required.")
    except Exception as e: # Catches errors from urlparse itself for malformed URLs
        raise ValueError(f"Invalid URL: {url}. Parsing failed: {e}")

def construct_resource_path(resource_id: str, resource_type: str) -> str:
    """
    Constructs a resource path based on resource ID and type.

    Args:
        resource_id: The identifier of the resource.
        resource_type: The type of the resource (e.g., "images", "scripts").

    Returns:
        A string representing the resource path, e.g., "/resources/images/my_image.jpg".
    """
    if not isinstance(resource_id, str) or not isinstance(resource_type, str):
        raise TypeError("resource_id and resource_type must be strings.")
    if not resource_id.strip() or not resource_type.strip():
        raise ValueError("resource_id and resource_type cannot be empty or contain only whitespace.")
        
    # Path structure: /<resource_type_plural>/<resource_id>
    # Ensuring resource_type is plural and lowercase for consistency.
    # A more robust solution might involve a mapping or inflection library for pluralization.
    resource_type_processed = resource_type.lower().strip()
    if not resource_type_processed.endswith('s'):
        resource_type_plural = resource_type_processed + 's'
    else:
        resource_type_plural = resource_type_processed
        
    return f"/{resource_type_plural}/{resource_id.strip()}"

if __name__ == '__main__':
    # Example Usage
    valid_urls = [
        "http://www.example.com",
        "https://example.com/path?query=value#fragment",
        "ftp://user:password@host:port/path",
    ]
    invalid_urls = [
        "www.example.com",  # Missing scheme
        "http://",  # Missing netloc
        "just_a_string",
        "http:///a", # malformed
        None, # Will be caught by type hinting if used, but good to test
        123,
    ]

    print("Testing valid URLs:")
    for u in valid_urls:
        try:
            if validate_url(u):
                print(f"'{u}' is valid.")
        except ValueError as e:
            print(f"Error validating '{u}': {e}")

    print("\nTesting invalid URLs:")
    for u in invalid_urls:
        try:
            # Ensure `u` is a string for the function, as per its type hint
            if not isinstance(u, str):
                print(f"Skipping non-string input: {u}")
                # Or raise a TypeError here if strict input type is desired before calling validate_url
                # raise TypeError(f"URL must be a string, got {type(u)}")
                # For now, we'll let validate_url handle it or pass it as is if it's a string
                if validate_url(str(u) if u is not None else ""): # Example of converting, though not ideal
                     print(f"'{u}' is valid (after potential conversion).") # Should not happen for these test cases
            elif validate_url(u):
                print(f"'{u}' is valid.") # Should not happen for these test cases
        except ValueError as e:
            print(f"Error validating '{u}': {e}")
        except TypeError as e: # If we added a TypeError for non-string inputs
            print(f"Input error for '{u}': {e}")

    print("\nTesting construct_resource_path:")
    resource_examples = [
        ("image123.png", "image"),
        ("main_script.js", "script"),
        ("document_abc.pdf", "documents"),
        ("  user_avatar.jpg  ", "  Avatars  "), # Test stripping and case
    ]
    for res_id, res_type in resource_examples:
        try:
            path = construct_resource_path(res_id, res_type)
            print(f"Path for ID '{res_id}' (type '{res_type}'): {path}")
        except (ValueError, TypeError) as e:
            print(f"Error for ID '{res_id}', type '{res_type}': {e}")
    
    invalid_resource_inputs = [
        ("", "images"), 
        ("  ", "images"), # Whitespace only ID
        ("id1", ""),   
        ("id1", "  "), # Whitespace only type
        (123, "images"), 
        ("id1", 123),  
    ]
    print("\nTesting invalid inputs for construct_resource_path:")
    for res_id, res_type in invalid_resource_inputs:
        try:
            path = construct_resource_path(res_id, res_type)
            print(f"Path for ID '{res_id}' (type '{res_type}'): {path}") # Should not be reached
        except (ValueError, TypeError) as e:
            print(f"Error for ID '{res_id}', type '{res_type}': {e}")
