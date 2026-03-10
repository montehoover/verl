from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def construct_redirect_link(domain_base_url, next_redirect_param):
    # Construct the full redirect URL
    if next_redirect_param.startswith('/'):
        # Relative path - combine with base URL
        redirect_url = urlparse.urljoin(domain_base_url, next_redirect_param)
    else:
        # Absolute URL or full path
        redirect_url = next_redirect_param
    
    # Parse both URLs for comparison
    base_parsed = urlparse.urlparse(domain_base_url)
    redirect_parsed = urlparse.urlparse(redirect_url)
    
    # Check if redirect URL has a valid scheme
    if redirect_parsed.scheme and redirect_parsed.scheme not in ['http', 'https']:
        raise ValueError(f"Invalid URL scheme: {redirect_parsed.scheme}")
    
    # Check if redirect is within the base domain
    if redirect_parsed.netloc == base_parsed.netloc or not redirect_parsed.netloc:
        return redirect_url
    
    # Check if redirect is to an allowed external domain
    if redirect_parsed.netloc in ALLOWED_DOMAINS:
        return redirect_url
    
    # If neither condition is met, raise ValueError
    raise ValueError(f"Redirect URL {redirect_url} is not within the base domain {base_parsed.netloc} or in the allowed domains list")
