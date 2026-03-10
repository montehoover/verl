from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the URL components (scheme, hostname, path).
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "hostname": parsed_url.hostname,
        "path": parsed_url.path,
        "params": parsed_url.params,
        "query": parsed_url.query,
        "fragment": parsed_url.fragment,
        "port": parsed_url.port,
        "username": parsed_url.username,
        "password": parsed_url.password,
    }

if __name__ == '__main__':
    # Example usage:
    test_url_1 = "http://www.example.com/path/to/resource?query=value#fragment"
    parts_1 = extract_url_parts(test_url_1)
    print(f"URL: {test_url_1}")
    print(f"Scheme: {parts_1['scheme']}")
    print(f"Hostname: {parts_1['hostname']}")
    print(f"Path: {parts_1['path']}")
    print(f"Query: {parts_1['query']}")
    print(f"Fragment: {parts_1['fragment']}")
    print(f"Full parts: {parts_1}")
    print("-" * 20)

    test_url_2 = "https://user:password@subdomain.example.co.uk:8080/another/path.html?name=test&age=30#section2"
    parts_2 = extract_url_parts(test_url_2)
    print(f"URL: {test_url_2}")
    print(f"Scheme: {parts_2['scheme']}")
    print(f"Hostname: {parts_2['hostname']}")
    print(f"Path: {parts_2['path']}")
    print(f"Port: {parts_2['port']}")
    print(f"Username: {parts_2['username']}")
    # Password is not printed for security reasons in real applications
    # print(f"Password: {parts_2['password']}")
    print(f"Query: {parts_2['query']}")
    print(f"Fragment: {parts_2['fragment']}")
    print(f"Full parts: {parts_2}")
    print("-" * 20)

    test_url_3 = "ftp://ftp.example.com/files/archive.zip"
    parts_3 = extract_url_parts(test_url_3)
    print(f"URL: {test_url_3}")
    print(f"Full parts: {parts_3}")
    print("-" * 20)

    test_url_4 = "/relative/path/to/file.txt" # Relative URL
    parts_4 = extract_url_parts(test_url_4)
    print(f"URL: {test_url_4}")
    print(f"Full parts: {parts_4}")
    print("-" * 20)
