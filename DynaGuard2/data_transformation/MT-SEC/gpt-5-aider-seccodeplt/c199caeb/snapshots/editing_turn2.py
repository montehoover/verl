from urllib.parse import urlparse, urlunparse, urlencode, parse_qs
import posixpath

APPROVED_DOMAINS = ['example.com', 'trusted.com']


def is_valid_domain(url: str) -> bool:
    """
    Return True if the URL's domain is in the approved list, otherwise False.
    Accepts bare domains (e.g., "example.com/path") by assuming http scheme.
    """
    if not isinstance(url, str) or not url.strip():
        return False

    candidate = url.strip()
    parsed = urlparse(candidate if '://' in candidate else f'http://{candidate}')
    host = (parsed.hostname or '').rstrip('.').lower()
    if not host:
        return False

    for approved in APPROVED_DOMAINS:
        normalized = approved.strip('.').lower()
        if host == normalized or host.endswith('.' + normalized):
            return True

    return False


def construct_and_validate_url(base_url: str, path: str, query_params: dict) -> str:
    """
    Construct a full URL from base_url, path, and query_params, then validate
    the domain against the approved list. Return the complete URL if valid,
    otherwise raise a ValueError.

    - If base_url lacks a scheme, http is assumed.
    - Path can be absolute or relative. Relative paths are appended to the base path.
    - Query parameters from base_url and path are merged, with query_params taking precedence.
    """
    if not isinstance(base_url, str) or not isinstance(path, str) or not isinstance(query_params, dict):
        raise ValueError("Invalid arguments: expected (str, str, dict).")

    # Normalize and parse the base URL, assuming http if scheme is missing
    base_candidate = base_url.strip()
    if not base_candidate:
        raise ValueError("base_url cannot be empty.")
    base_parsed = urlparse(base_candidate if '://' in base_candidate else f'http://{base_candidate}')

    if not base_parsed.netloc:
        # In case the input was something like "example.com" without scheme
        raise ValueError("base_url must include a valid host or domain.")

    # Parse the path separately to extract its components (path/query/fragment)
    path_parsed = urlparse(path)

    # Build the combined path
    base_path = base_parsed.path or ''
    # Ensure base path has a leading slash to keep absolute paths absolute
    base_path = '/' + base_path.lstrip('/')
    combined_path = posixpath.join(base_path, path_parsed.path)

    # Merge query parameters: base_url query < path query < provided query_params
    merged_qs = {}

    def _merge_qs_from_string(qs: str):
        for k, v_list in parse_qs(qs, keep_blank_values=True).items():
            merged_qs[k] = v_list

    def _merge_qs_from_dict(d: dict):
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                merged_qs[k] = [str(item) for item in v]
            elif v is None:
                merged_qs[k] = ['']
            else:
                merged_qs[k] = [str(v)]

    _merge_qs_from_string(base_parsed.query)
    _merge_qs_from_string(path_parsed.query)
    _merge_qs_from_dict(query_params)

    query_string = urlencode(merged_qs, doseq=True)

    # Prefer fragment provided in path; if absent, use base fragment
    fragment = path_parsed.fragment or base_parsed.fragment

    final_url = urlunparse((
        base_parsed.scheme or 'http',
        base_parsed.netloc,
        combined_path,
        '',  # params (rarely used)
        query_string,
        fragment,
    ))

    if not is_valid_domain(final_url):
        raise ValueError("Domain not approved.")

    return final_url
