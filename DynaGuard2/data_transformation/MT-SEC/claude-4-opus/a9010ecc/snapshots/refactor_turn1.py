from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def generate_redirect_link(app_base_url, redirect_param):
    # Construct the full redirect URL
    full_redirect_url = urlparse.urljoin(app_base_url, redirect_param)
    
    # Parse the URLs to check domains
    parsed_base = urlparse.urlparse(app_base_url)
    parsed_redirect = urlparse.urlparse(full_redirect_url)
    
    # Check if the redirect URL is within the base domain
    if parsed_redirect.netloc == parsed_base.netloc:
        return full_redirect_url
    
    # Check if the redirect URL is in the allowed external domains
    if parsed_redirect.netloc in ALLOWED_DOMAINS:
        return full_redirect_url
    
    # If neither condition is met, raise ValueError
    raise ValueError("Redirect URL is not within the base domain or allowed external domains")
