from urllib.parse import urlparse, urlunparse, urljoin, parse_qsl, urlencode

ALLOWED_DOMAINS = ['example.com', 'another-example.com']


def is_valid_domain(url: str) -> bool:
    if not isinstance(url, str):
        return False

    url = url.strip()
    if not url:
        return False

    parsed = urlparse(url)
    # If no scheme is present, urlparse puts the host in path; prepend a scheme to parse correctly.
    if not parsed.netloc:
        parsed = urlparse('http://' + url)

    host = parsed.hostname
    if not host:
        return False

    host = host.rstrip('.').lower()
    return host in ALLOWED_DOMAINS


def construct_and_validate_url(base_url: str, path: str, query_params: dict) -> str:
    if not isinstance(base_url, str) or not isinstance(path, str):
        raise ValueError("base_url and path must be strings")
    if query_params is None:
        query_params = {}
    if not isinstance(query_params, dict):
        raise ValueError("query_params must be a dictionary")

    base_url = base_url.strip()
    path = path.strip()

    # Parse and normalize the base URL; add a default scheme if missing
    parsed_base = urlparse(base_url)
    if not parsed_base.netloc:
        parsed_base = urlparse('http://' + base_url)

    # Sanitize the provided path to avoid overriding the base (ignore scheme/netloc in path)
    sanitized_path = urlparse(path).path if path else ''

    # Prepare a base URL without query/fragment for joining paths
    base_for_join = urlunparse((
        parsed_base.scheme or 'http',
        parsed_base.netloc,
        parsed_base.path or '',
        '', '', ''
    ))

    # Join the path to the base URL
    joined = urljoin(base_for_join, sanitized_path)
    parsed_joined = urlparse(joined)

    # Merge existing query from the base with the provided query_params
    existing_pairs = parse_qsl(parsed_joined.query, keep_blank_values=True)
    new_qs = urlencode(query_params, doseq=True)
    new_pairs = parse_qsl(new_qs, keep_blank_values=True) if new_qs else []
    combined_pairs = existing_pairs + new_pairs
    final_query = urlencode(combined_pairs, doseq=True)

    final_url = urlunparse(parsed_joined._replace(query=final_query))

    if not is_valid_domain(final_url):
        raise ValueError("URL domain is not allowed")

    return final_url
