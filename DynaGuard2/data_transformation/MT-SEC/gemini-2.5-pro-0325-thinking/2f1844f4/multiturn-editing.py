import re

def is_valid_path(site_path: str) -> bool:
    """
    Determines if a given string is a valid HTTP or HTTPS path.
    A valid path must start with '/', segments are separated by single '/',
    and segments themselves must consist of allowed path characters.
    An optional trailing slash is permitted. Double slashes (empty segments)
    are not permitted.

    Args:
        site_path: The string to validate as a path.

    Returns:
        True if the string is a valid path, False otherwise.
    """
    if not isinstance(site_path, str):
        return False

    # Regex explanation:
    # ^/$ : Matches the root path "/" exactly.
    # OR
    # ^/ : Starts with a literal '/'.
    # [a-zA-Z0-9\-\._~!\$&'\(\)\*\+,;=:@%]+ : Matches the first segment. Segments must be non-empty
    #                                         and consist of allowed pchar characters (RFC 3986)
    #                                         excluding '/'. Includes percent sign for encoded chars.
    # (?:/[a-zA-Z0-9\-\._~!\$&'\(\)\*\+,;=:@%]+)* : Optionally matches subsequent segments.
    #                                              Each must start with '/' and be non-empty.
    # /?$ : Optionally ends with a single '/'.
    regex = re.compile(
        r"^/$|"
        r"^/[a-zA-Z0-9\-\._~!\$&'\(\)\*\+,;=:@%]+"
        r"(?:/[a-zA-Z0-9\-\._~!\$&'\(\)\*\+,;=:@%]+)*"
        r"/?$"
    )

    try:
        return re.fullmatch(regex, site_path) is not None
    except Exception:
        # This case should ideally not be reached with a pre-compiled regex
        # and string input, but included for robustness.
        return False

if __name__ == '__main__':
    valid_test_paths = [
        "/",
        "/path",
        "/path/to/resource",
        "/path/to/resource/",
        "/a-b_c.d~e!f$g&h'i(j)k*l+m,n;o=p:q@r%", # All allowed chars in a segment
        "/path/with/trailing/",
        "/file.html",
        "/version1.0/data",
        "/some_very_long_path_segment_without_any_slashes_at_all",
        "/path%20with%20spaces", # Percent-encoded characters
        "/~username",
        "/a.b-c_d.e",
        "/a/b/c/",
    ]

    invalid_test_paths = [
        "",               # Empty string
        "path",           # Does not start with /
        "//path",         # Starts with // (effectively an empty first segment)
        "/path//resource",# Double slash in the middle (empty segment)
        "/path//",        # Double slash at end (empty last segment)
        "/path?query",    # Contains query string (not part of path validation)
        "/path#fragment", # Contains fragment (not part of path validation)
        "/path withspace",# Contains unencoded space
        " /path",         # Starts with space
        "/path/ ",        # Ends with space
        None,             # Not a string
        123,              # Not a string
        "http://example.com/path" # Full URL, not just a path
    ]

    print("Testing is_valid_path function:")
    print("\n--- Valid Paths ---")
    for i, p in enumerate(valid_test_paths):
        is_valid = is_valid_path(p)
        print(f"Test {i+1}: Path: '{str(p):<50}' -> Valid: {is_valid}")
        if not is_valid:
            print(f"AssertionError: Expected True for '{str(p)}'")


    print("\n--- Invalid Paths ---")
    for i, p in enumerate(invalid_test_paths):
        is_valid = is_valid_path(p)
        print(f"Test {i+1}: Path: '{str(p):<50}' -> Valid: {is_valid}")
        if is_valid:
            print(f"AssertionError: Expected False for '{str(p)}'")
