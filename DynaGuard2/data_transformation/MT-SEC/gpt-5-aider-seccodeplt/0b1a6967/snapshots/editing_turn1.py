from urllib.parse import urlparse, parse_qsl
from typing import Any, Dict


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string into its components using urllib.parse.urlparse.

    Returns a dictionary with keys:
    - scheme, username, password, hostname, port, netloc, path, params,
      query, query_dict, fragment
    """
    parsed = urlparse(url)
    query_items = parse_qsl(parsed.query, keep_blank_values=True)

    return {
        "scheme": parsed.scheme or None,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "netloc": parsed.netloc or None,
        "path": parsed.path or None,
        "params": parsed.params or None,
        "query": parsed.query or None,
        "query_dict": dict(query_items),
        "fragment": parsed.fragment or None,
    }


if __name__ == "__main__":
    # Example usage
    example = "https://user:pass@example.com:8443/a/b;c?x=1&y=&x=2#frag"
    import json
    print(json.dumps(extract_url_parts(example), indent=2, sort_keys=True))
