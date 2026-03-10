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

def construct_external_resource_url(cdn_base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Constructs a complete URL for an external resource on a CDN.

    Args:
        cdn_base_url: The base URL of the CDN (e.g., "https://cdn.example.com").
        resource_id: The identifier of the resource.
        resource_type: The type of the resource.

    Returns:
        A string representing the complete URL to the resource.

    Raises:
        ValueError: If the cdn_base_url is invalid, not from an allowed domain,
                    or if resource_id/resource_type are invalid.
        TypeError: If any of the inputs are not strings.
    """
    if not isinstance(cdn_base_url, str) or \
       not isinstance(resource_id, str) or \
       not isinstance(resource_type, str):
        raise TypeError("cdn_base_url, resource_id, and resource_type must be strings.")

    if not cdn_base_url.strip():
        raise ValueError("cdn_base_url cannot be empty or contain only whitespace.")

    # Validate the cdn_base_url structure and ensure it's from an allowed domain
    try:
        # First, ensure it's a generally valid URL format
        if not validate_url(cdn_base_url): 
            # validate_url itself raises ValueError for specific issues, 
            # but we catch its ValueError to provide a more specific one if needed,
            # or rely on its message. Here, we'll let its error propagate if it's a format issue.
            pass # Should have been caught by validate_url if malformed

        parsed_cdn_url = urlparse(cdn_base_url)
        if not parsed_cdn_url.netloc: # Should be caught by validate_url, but double check
             raise ValueError(f"Invalid CDN base URL: {cdn_base_url}. Missing network location.")
        if parsed_cdn_url.netloc not in ALLOWED_CDN_DOMAINS:
            raise ValueError(f"CDN domain '{parsed_cdn_url.netloc}' is not allowed.")
    except ValueError as e: # Catch ValueError from validate_url or our own checks
        # Re-raise with more context or let original propagate
        raise ValueError(f"Invalid CDN base URL '{cdn_base_url}': {e}")


    # Construct the path part of the URL
    # This will raise ValueError/TypeError for invalid resource_id/type
    resource_path = construct_resource_path(resource_id, resource_type)

    # Ensure resource_path starts with a slash if urljoin needs it,
    # or handle base_url not ending with a slash.
    # urljoin handles this well:
    # urljoin("http://cdn.com/foo", "/bar/baz.jpg") -> "http://cdn.com/bar/baz.jpg"
    # urljoin("http://cdn.com/foo/", "bar/baz.jpg") -> "http://cdn.com/foo/bar/baz.jpg"
    # Our construct_resource_path already returns a path starting with "/"
    
    complete_url = urljoin(cdn_base_url, resource_path.lstrip('/')) # lstrip '/' if base_url might have trailing path
    
    # A more robust urljoin: ensure cdn_base_url ends with '/' if it has no path component
    # to avoid issues like urljoin('http://host', 'path') -> 'http://path'
    # However, our validate_url ensures scheme and netloc, and urljoin is generally smart.
    # For example: urljoin('https://cdn.example.com', 'images/pic.jpg')
    # If cdn_base_url is "https://cdn.example.com", and resource_path is "/images/img.png"
    # urljoin("https://cdn.example.com", "/images/img.png") -> "https://cdn.example.com/images/img.png"
    # If cdn_base_url is "https://cdn.example.com/static/", and resource_path is "/images/img.png"
    # urljoin("https://cdn.example.com/static/", "/images/img.png") -> "https://cdn.example.com/images/img.png" (path replaced)
    # To append, resource_path should not be absolute:
    # urljoin("https://cdn.example.com/static/", "images/img.png") -> "https://cdn.example.com/static/images/img.png"
    # Our construct_resource_path returns an absolute path like "/images/id.jpg".
    # So, we should use it carefully with urljoin.
    # A common pattern is base_url + path_segment.
    # If cdn_base_url is "https://cdn.example.com" and path is "/images/foo.jpg"
    # We want "https://cdn.example.com/images/foo.jpg"
    
    # Let's refine the join:
    # Ensure base_url does not have a trailing slash if path starts with one.
    # Or ensure path does not start with one if base_url has no path or ends with slash.
    
    # urlparse(cdn_base_url)._replace(path=resource_path).geturl() is an option if we want to replace the path
    # but we want to append to the base.
    
    # The most straightforward way with urljoin, given our resource_path starts with '/':
    # If cdn_base_url = "https://cdn.example.com", path = "/scripts/main.js"
    # urljoin("https://cdn.example.com", "/scripts/main.js") -> "https://cdn.example.com/scripts/main.js" (Correct)
    # If cdn_base_url = "https://cdn.example.com/some/base/", path = "/scripts/main.js"
    # urljoin("https://cdn.example.com/some/base/", "/scripts/main.js") -> "https://cdn.example.com/scripts/main.js" (Path replaced, might be desired)
    # If we want to append to "https://cdn.example.com/some/base/", path should be relative: "scripts/main.js"
    # Our `construct_resource_path` returns an absolute-like path.
    # For CDN, usually the base URL is just the domain or a top-level path.
    
    # Assuming cdn_base_url is like "https://cdn.example.com" or "https://cdn.example.com/static"
    # and resource_path is like "/images/foo.jpg"
    
    final_url = urljoin(cdn_base_url.rstrip('/') + '/', resource_path.lstrip('/'))
    # Example:
    # cdn_base_url = "https://cdn.example.com", resource_path = "/images/foo.jpg"
    # "https://cdn.example.com/" + "images/foo.jpg" -> "https://cdn.example.com/images/foo.jpg"
    # cdn_base_url = "https://cdn.example.com/static", resource_path = "/images/foo.jpg"
    # "https://cdn.example.com/static/" + "images/foo.jpg" -> "https://cdn.example.com/static/images/foo.jpg"
    
    return final_url


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

    print("\nTesting construct_external_resource_url:")
    cdn_test_cases = [
        ("https://cdn.example.com", "img.jpg", "image", True),
        ("http://assets.example.net/static/", "script.js", "scripts", True),
        ("https://static.example.org", "data.json", "files", True),
        ("https://unallowed.com", "img.jpg", "image", False), # Not allowed domain
        ("cdn.example.com", "img.jpg", "image", False), # Invalid base URL (no scheme)
        ("ftp://cdn.example.com", "img.jpg", "image", True), # Allowed domain, ftp scheme (validate_url allows it)
        ("https://cdn.example.com", "", "image", False), # Invalid resource_id
        ("https://cdn.example.com", "img.jpg", "", False), # Invalid resource_type
        (123, "img.jpg", "image", False), # Invalid base_url type
    ]

    for base, res_id, res_type, should_succeed in cdn_test_cases:
        try:
            url = construct_external_resource_url(base, res_id, res_type)
            if should_succeed:
                print(f"Generated URL for '{base}', '{res_id}', '{res_type}': {url}")
            else:
                print(f"ERROR: Test case for '{base}', '{res_id}', '{res_type}' should have failed but returned: {url}")
        except (ValueError, TypeError) as e:
            if should_succeed:
                print(f"ERROR: Test case for '{base}', '{res_id}', '{res_type}' failed unexpectedly: {e}")
            else:
                print(f"Successfully caught error for '{base}', '{res_id}', '{res_type}': {e}")
    
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
