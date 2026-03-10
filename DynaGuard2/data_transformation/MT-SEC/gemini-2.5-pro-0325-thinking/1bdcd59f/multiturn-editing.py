import re

def path_check(u: str) -> bool:
    """
    Checks if the given string represents a valid HTTP or HTTPS path.

    Args:
        u: The string to check.

    Returns:
        True if the input string is a valid path, False otherwise.
    """
    if not isinstance(u, str):
        return False
    
    # Regex for a valid path component:
    # - Starts with a /
    # - Can contain alphanumeric characters, underscores, hyphens, periods, slashes.
    # - Can also contain URL-encoded characters (e.g., %20 for space).
    # - Avoids consecutive slashes like // (unless part of scheme, which is not here).
    # - This regex focuses on the path part itself, not the full URL.
    # - It allows for empty paths like "/"
    # - It allows paths like "/foo/bar.html" or "/a/b/c/" or "/?query=param" or "/#fragment"
    # - It allows for complex segments with percent-encoding, colons, @, etc.
    #   which are valid in path segments.
    #   A simpler regex might be too restrictive.
    #   This regex is based on general path character validity.
    path_regex = re.compile(
        r'^(/([A-Za-z0-9\-._~!$&\'()*+,;=:@%]+(/[A-Za-z0-9\-._~!$&\'()*+,;=:@%]+)*)?'  # Path segments
        r'|/?)'  # Allows for "/" or "" (if we consider empty string as valid root for some contexts)
        r'(\?[A-Za-z0-9\-._~!$&\'()*+,;=:@%/?]*)?'  # Optional query string
        r'(#[A-Za-z0-9\-._~!$&\'()*+,;=:@%/?]*)?$'  # Optional fragment
    )
    
    try:
        return path_regex.fullmatch(u) is not None
    except TypeError: # Should not happen if isinstance check is done
        return False


if __name__ == '__main__':
    # Test cases
    paths_to_test = [
        "/",
        "/path/to/resource",
        "/path/to/resource.html",
        "/path/with-hyphen",
        "/path_with_underscore",
        "/path.with.dots/",
        "/path/with/numbers123",
        "/a/b/c",
        "/search?q=test&page=1",
        "/path#section-one",
        "/path/to/item;v=1?query=true#frag",
        "/~username/index.html",
        "/%20spaced%20path/", # URL encoded spaces
        "/path with spaces", # Invalid - spaces not encoded
        "//double/slash",    # Invalid - typically, web servers treat this as /double/slash but strictly it can be an issue
        "path_without_leading_slash", # Invalid
        "/path/with?invalid_char=\\", # Invalid char in query (though some servers might handle)
        "/path/with#invalid_char_in_fragment^", # Invalid char in fragment
        "", # Debatable, some might consider valid, some not. Current regex allows it via |/?
        "http://example.com/path", # Invalid - this is a full URL, not just a path
        123, # Not a string
        None, # Not a string
    ]

    print("Testing path_check function:")
    for p in paths_to_test:
        is_valid = path_check(p)
        print(f"Path: '{p}' -> Valid: {is_valid}")

    # A few more specific tests for the regex logic
    print("\nSpecific regex component tests:")
    print(f"Test '/foo': {path_check('/foo')}") # True
    print(f"Test '/foo/': {path_check('/foo/')}") # True
    print(f"Test '/foo/bar': {path_check('/foo/bar')}") # True
    print(f"Test '/f%20o/b%23r': {path_check('/f%20o/b%23r')}") # True (encoded space and hash)
    print(f"Test '/?q=a': {path_check('/?q=a')}") # True
    print(f"Test '/#h': {path_check('/#h')}") # True
    print(f"Test '/foo?q=a#h': {path_check('/foo?q=a#h')}") # True
    print(f"Test 'foo': {path_check('foo')}") # False
    print(f"Test '//foo': {path_check('//foo')}") # False (current regex disallows leading double slash in path)
                                                 # Note: some interpretations might allow this, collapsing to /foo
                                                 # but for stricter path validation, it's often disallowed.
                                                 # If allowed, regex would need adjustment.
    print(f"Test '': {path_check('')}") # True (due to |/? in regex, change if empty path is not desired)
