from urllib.parse import urlencode, urljoin

def build_url_with_query(base_url, path, query_params):
    # Join base URL and path
    url = urljoin(base_url, path)
    
    # Add query parameters if any
    if query_params:
        query_string = urlencode(query_params)
        url = f"{url}?{query_string}"
    
    return url
