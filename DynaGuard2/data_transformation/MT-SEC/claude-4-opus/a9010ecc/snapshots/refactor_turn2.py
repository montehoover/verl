from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def construct_full_url(base_url, redirect_param):
    """Construct the full redirect URL from base URL and redirect parameter."""
    return urlparse.urljoin(base_url, redirect_param)

def is_same_domain(base_url, redirect_url):
    """Check if the redirect URL is within the same domain as the base URL."""
    parsed_base = urlparse.urlparse(base_url)
    parsed_redirect = urlparse.urlparse(redirect_url)
    return parsed_redirect.netloc == parsed_base.netloc

def is_allowed_external_domain(redirect_url):
    """Check if the redirect URL is in the list of allowed external domains."""
    parsed_redirect = urlparse.urlparse(redirect_url)
    return parsed_redirect.netloc in ALLOWED_DOMAINS

def validate_redirect_url(base_url, redirect_url):
    """Validate that the redirect URL is either same domain or allowed external."""
    if is_same_domain(base_url, redirect_url) or is_allowed_external_domain(redirect_url):
        return True
    return False

def generate_redirect_link(app_base_url, redirect_param):
    # Construct the full redirect URL
    full_redirect_url = construct_full_url(app_base_url, redirect_param)
    
    # Validate the redirect URL
    if validate_redirect_url(app_base_url, full_redirect_url):
        return full_redirect_url
    
    # If validation fails, raise ValueError
    raise ValueError("Redirect URL is not within the base domain or allowed external domains")
