from urllib import parse as urlparse
from typing import Optional

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _normalize_hostname(host: Optional[str]) -> Optional[str]:
    if not host:
        return None
    return host.rstrip('.').lower()


def is_valid_domain(url: str) -> bool:
    if not isinstance(url, str):
        return False

    url = url.strip()
    if not url:
        return False

    parsed = urlparse.urlparse(url)
    # If no scheme is present, urlparse puts the host in path; prepend a scheme to parse correctly.
    if not parsed.netloc:
        parsed = urlparse.urlparse('http://' + url)

    host = _normalize_hostname(parsed.hostname)
    if not host:
        return False

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
    parsed_base = urlparse.urlparse(base_url)
    if not parsed_base.netloc:
        parsed_base = urlparse.urlparse('http://' + base_url)

    # Sanitize the provided path to avoid overriding the base (ignore scheme/netloc in path)
    sanitized_path = urlparse.urlparse(path).path if path else ''

    # Prepare a base URL without query/fragment for joining paths
    base_for_join = urlparse.urlunparse((
        parsed_base.scheme or 'http',
        parsed_base.netloc,
        parsed_base.path or '',
        '', '', ''
    ))

    # Join the path to the base URL
    joined = urlparse.urljoin(base_for_join, sanitized_path)
    parsed_joined = urlparse.urlparse(joined)

    # Merge existing query from the base with the provided query_params
    existing_pairs = urlparse.parse_qsl(parsed_joined.query, keep_blank_values=True)
    new_qs = urlparse.urlencode(query_params, doseq=True)
    new_pairs = urlparse.parse_qsl(new_qs, keep_blank_values=True) if new_qs else []
    combined_pairs = existing_pairs + new_pairs
    final_query = urlparse.urlencode(combined_pairs, doseq=True)

    final_url = urlparse.urlunparse(parsed_joined._replace(query=final_query))

    if not is_valid_domain(final_url):
        raise ValueError("URL domain is not allowed")

    return final_url


def construct_redirect_link(domain_base_url: str, next_redirect_param: str) -> str:
    if not isinstance(domain_base_url, str) or not isinstance(next_redirect_param, str):
        raise ValueError("domain_base_url and next_redirect_param must be strings")

    base_input = domain_base_url.strip()
    if not base_input:
        raise ValueError("domain_base_url cannot be empty")

    # Normalize and parse the base domain URL; ensure there is a scheme to get hostname reliably
    parsed_base = urlparse.urlparse(base_input)
    if not parsed_base.netloc:
        parsed_base = urlparse.urlparse('http://' + base_input)

    base_scheme = parsed_base.scheme or 'http'
    base_netloc = parsed_base.netloc
    base_host = _normalize_hostname(parsed_base.hostname)
    if not base_host:
        raise ValueError("Invalid base URL")

    # Normalize the next parameter
    next_input = next_redirect_param.strip()

    # Default to root if empty
    if not next_input:
        next_input = '/'

    # Determine the absolute target URL
    next_parsed = urlparse.urlparse(next_input)
    if next_parsed.netloc:
        # Network-path reference (e.g., //host/path) or absolute with netloc but maybe no scheme.
        target_url = urlparse.urlunparse((
            base_scheme if not next_parsed.scheme else next_parsed.scheme,
            next_parsed.netloc,
            next_parsed.path or '',
            next_parsed.params,
            next_parsed.query,
            next_parsed.fragment
        ))
    elif next_parsed.scheme and not next_parsed.netloc:
        # Schemes without netloc (e.g., mailto:, javascript:) are not allowed
        raise ValueError("Invalid redirect URL scheme")
    elif next_parsed.scheme and next_parsed.netloc:
        # Absolute URL with its own scheme and netloc
        target_url = next_input
    else:
        # Relative path; join with base
        base_for_join = urlparse.urlunparse((
            base_scheme,
            base_netloc,
            parsed_base.path or '/',
            '', '', ''
        ))
        target_url = urlparse.urljoin(base_for_join, next_input)

    # Validate the target URL domain
    parsed_target = urlparse.urlparse(target_url)
    target_host = _normalize_hostname(parsed_target.hostname)

    if not target_host:
        raise ValueError("Invalid redirect URL")

    # Allow if same base host
    if target_host == base_host:
        return target_url

    # Allow if external domain is in the allowed list
    if target_host in ALLOWED_DOMAINS:
        return target_url

    # Otherwise, reject
    raise ValueError("Redirect URL domain is not allowed")
