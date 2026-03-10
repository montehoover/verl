def build_url_with_query(base_url: str, path: str, query_param: str) -> str:
    """
    Construct a URL by combining base_url and path, and append query_param.
    If the resulting URL already contains a query string, the query_param will be appended with '&',
    otherwise it will be introduced with '?'.
    """
    base = base_url.rstrip('/')
    if path.startswith('/'):
        joined = f"{base}{path}"
    elif path:
        joined = f"{base}/{path}"
    else:
        joined = base

    if query_param:
        sep = '&' if '?' in joined else '?'
        return f"{joined}{sep}{query_param}"
    return joined
