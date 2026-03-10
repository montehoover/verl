from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the scheme, hostname, and path of the URL.
        Returns None if the URL cannot be parsed.
    """
    try:
        parsed_url = urlparse(url_string)
        return {
            "scheme": parsed_url.scheme,
            "hostname": parsed_url.hostname,
            "path": parsed_url.path,
            "params": parsed_url.params,
            "query": parsed_url.query,
            "fragment": parsed_url.fragment,
            "port": parsed_url.port,
        }
    except Exception: # pylint: disable=broad-except
        # urlparse can raise various exceptions for malformed URLs,
        # though it's generally robust. Returning None for simplicity.
        return None

if __name__ == '__main__':
    test_url_1 = "http://www.example.com/path/to/resource?name=test#fragment"
    parts_1 = extract_url_parts(test_url_1)
    print(f"Parts for {test_url_1}: {parts_1}")

    test_url_2 = "https://subdomain.example.co.uk:8080/another/path.html?key1=value1&key2=value2"
    parts_2 = extract_url_parts(test_url_2)
    print(f"Parts for {test_url_2}: {parts_2}")

    test_url_3 = "ftp://user:password@ftp.example.com/files/archive.zip"
    parts_3 = extract_url_parts(test_url_3)
    print(f"Parts for {test_url_3}: {parts_3}")

    # Example of a URL without a path
    test_url_4 = "mailto:user@example.com"
    parts_4 = extract_url_parts(test_url_4)
    print(f"Parts for {test_url_4}: {parts_4}")

    # Example of a relative URL (hostname will be None)
    test_url_5 = "/relative/path?query=yes"
    parts_5 = extract_url_parts(test_url_5)
    print(f"Parts for {test_url_5}: {parts_5}")
    
    # Example of an invalid URL (might not raise an error but return empty parts)
    test_url_6 = "this is not a url"
    parts_6 = extract_url_parts(test_url_6)
    print(f"Parts for {test_url_6}: {parts_6}")
