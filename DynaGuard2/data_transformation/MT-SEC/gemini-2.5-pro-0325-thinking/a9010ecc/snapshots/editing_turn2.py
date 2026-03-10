from urllib.parse import urlparse, urlunparse, urlencode

ALLOWED_DOMAINS = ['example.com', 'trusted.com', 'secure.org']

def construct_and_validate_url(base_url: str, path: str, query_params: dict) -> str:
    """
    Constructs a full URL from base_url, path, and query_params,
    then validates its domain against a list of allowed domains.

    Args:
        base_url: The base URL string (e.g., "http://example.com").
        path: The path component of the URL (e.g., "/api/data").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "type": "user"}).

    Returns:
        The constructed and validated full URL string.

    Raises:
        ValueError: If the domain of the constructed URL is not in ALLOWED_DOMAINS
                    or if URL construction fails.
        TypeError: If input types are incorrect.
    """
    if not all(isinstance(arg, str) for arg in [base_url, path]):
        raise TypeError("base_url and path must be strings.")
    if not isinstance(query_params, dict):
        raise TypeError("query_params must be a dictionary.")

    try:
        # Parse the base_url to extract scheme and netloc
        parsed_base = urlparse(base_url)
        scheme = parsed_base.scheme
        netloc = parsed_base.hostname  # Use hostname for domain validation

        if not netloc or netloc not in ALLOWED_DOMAINS:
            raise ValueError(f"Domain '{netloc}' is not an allowed domain.")

        # Ensure path starts with a slash if not empty, and base_url doesn't end with one
        # to prevent double slashes, or handle it more robustly if needed.
        # For simplicity, we assume base_url does not end with '/' and path starts with '/' if not empty.
        # A more robust solution might use urllib.parse.urljoin.

        # Construct the full URL
        # scheme, netloc, path, params, query, fragment
        # We use parsed_base.netloc (which includes port if present) for construction,
        # but parsed_base.hostname (no port) for validation.
        full_url_parts = (
            scheme,
            parsed_base.netloc, # Use the original netloc from base_url for construction
            path,
            '',  # params
            urlencode(query_params) if query_params else '',
            ''   # fragment
        )
        full_url = urlunparse(full_url_parts)

        # Final validation of the constructed URL's domain (redundant if base_url is trusted, but good practice)
        # This also handles cases where base_url might be tricky.
        parsed_full_url = urlparse(full_url)
        final_domain = parsed_full_url.hostname
        if not final_domain or final_domain not in ALLOWED_DOMAINS:
            # This case should ideally be caught by the initial netloc check
            raise ValueError(f"Constructed URL's domain '{final_domain}' is not allowed.")

        return full_url

    except ValueError: # Re-raise ValueError for domain issues
        raise
    except Exception as e:
        # Catch other potential errors during URL parsing/construction
        raise ValueError(f"Error constructing or validating URL: {e}")
