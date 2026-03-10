from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


def build_url_with_query(base_url: str, path: str, query_params: dict) -> str:
    """
    Build a complete URL by combining a base URL, a path, and query parameters.

    - If `path` starts with '/', it is treated as absolute relative to the base domain.
    - Otherwise, it is appended to the base URL's path with a single slash separator.
    - Existing query parameters in `base_url` or `path` are preserved, but any keys
      provided in `query_params` override them. Values of None are omitted.
    - Lists/tuples in `query_params` produce repeated keys in the query string (doseq=True).
    """
    base = urlparse(base_url if base_url is not None else "")
    path = path or ""

    # Parse any query/fragment contained in the path argument itself
    path_parts = urlparse(path)

    # Resolve the final path
    if path_parts.path:
        if path_parts.path.startswith('/'):
            final_path = path_parts.path
        else:
            base_path = base.path or ""
            if base_path.endswith('/'):
                final_path = base_path + path_parts.path
            elif base_path:
                final_path = base_path + '/' + path_parts.path
            else:
                final_path = '/' + path_parts.path if base.netloc else path_parts.path
    else:
        final_path = base.path

    # Merge query parameters: base -> path -> provided query_params (override)
    merged = {}

    # Start with existing query parameters from base and path
    for k, v in parse_qsl(base.query, keep_blank_values=True):
        merged.setdefault(k, []).append(v)
    for k, v in parse_qsl(path_parts.query, keep_blank_values=True):
        merged.setdefault(k, []).append(v)

    # Apply provided query_params with override semantics
    if query_params:
        for k, v in query_params.items():
            if v is None:
                # Skip None values
                continue
            if isinstance(v, (list, tuple)):
                values = [str(x) for x in v if x is not None]
                if not values:
                    continue
                merged[k] = values
            else:
                merged[k] = [str(v)]

    # Build the query string
    query_string = urlencode(merged, doseq=True)

    # Prefer fragment from path if provided; otherwise keep base fragment
    fragment = path_parts.fragment or base.fragment

    result = base._replace(path=final_path, query=query_string, fragment=fragment)
    return urlunparse(result)
