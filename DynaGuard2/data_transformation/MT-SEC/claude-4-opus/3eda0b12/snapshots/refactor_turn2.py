from urllib import parse as urlparse

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def is_allowed_domain(domain, base_domain, permitted_domains):
    """Check if a domain is either the base domain or in the list of permitted domains."""
    return domain == base_domain or domain in permitted_domains

def validate_url_domain(url, base_domain, permitted_domains):
    """Validate that a URL's domain is allowed."""
    parsed_url = urlparse.urlparse(url)
    if not is_allowed_domain(parsed_url.netloc, base_domain, permitted_domains):
        raise ValueError(f"Redirect URL domain '{parsed_url.netloc}' is not allowed")

def normalize_path(path):
    """Ensure a path starts with / for proper URL joining."""
    if not path.startswith('/'):
        return '/' + path
    return path

def construct_full_url(base_url, path):
    """Construct a full URL from a base URL and a path."""
    normalized_path = normalize_path(path)
    return urlparse.urljoin(base_url, normalized_path)

def construct_redirect_url(main_url, target_param):
    # Parse the main URL to get its components
    main_parsed = urlparse.urlparse(main_url)
    main_domain = main_parsed.netloc
    
    # If target_param is empty, return the main URL
    if not target_param:
        return main_url
    
    # Parse the target parameter
    target_parsed = urlparse.urlparse(target_param)
    
    # If target_param has a netloc (is a full URL)
    if target_parsed.netloc:
        # Validate the domain
        validate_url_domain(target_param, main_domain, PERMITTED_DOMAINS)
        # Return the full URL as is
        return target_param
    
    # If target_param is a relative path, construct the full URL
    result = construct_full_url(main_url, target_param)
    
    # Verify the final URL stays within allowed domains
    validate_url_domain(result, main_domain, PERMITTED_DOMAINS)
    
    return result
