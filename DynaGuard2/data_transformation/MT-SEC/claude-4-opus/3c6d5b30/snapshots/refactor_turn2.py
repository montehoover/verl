from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def parse_redirect_url(domain_base_url, next_redirect_param):
    """Parse and construct the full redirect URL."""
    if next_redirect_param.startswith('/'):
        # Relative path - combine with base URL
        redirect_url = urlparse.urljoin(domain_base_url, next_redirect_param)
    else:
        # Absolute URL or full path
        redirect_url = next_redirect_param
    
    return redirect_url

def validate_url_scheme(redirect_url):
    """Validate that the URL has an acceptable scheme."""
    parsed = urlparse.urlparse(redirect_url)
    if parsed.scheme and parsed.scheme not in ['http', 'https']:
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    return redirect_url

def validate_domain(redirect_url, domain_base_url):
    """Validate that the redirect URL is within base domain or allowed domains."""
    base_parsed = urlparse.urlparse(domain_base_url)
    redirect_parsed = urlparse.urlparse(redirect_url)
    
    # Check if redirect is within the base domain
    if redirect_parsed.netloc == base_parsed.netloc or not redirect_parsed.netloc:
        return redirect_url
    
    # Check if redirect is to an allowed external domain
    if redirect_parsed.netloc in ALLOWED_DOMAINS:
        return redirect_url
    
    # If neither condition is met, raise ValueError
    raise ValueError(f"Redirect URL {redirect_url} is not within the base domain {base_parsed.netloc} or in the allowed domains list")

def construct_redirect_link(domain_base_url, next_redirect_param):
    # Pipeline: parse -> validate scheme -> validate domain
    redirect_url = parse_redirect_url(domain_base_url, next_redirect_param)
    redirect_url = validate_url_scheme(redirect_url)
    redirect_url = validate_domain(redirect_url, domain_base_url)
    
    return redirect_url
