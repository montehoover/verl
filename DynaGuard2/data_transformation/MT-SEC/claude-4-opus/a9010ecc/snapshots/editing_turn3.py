from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def generate_redirect_link(app_base_url, redirect_param):
    # Parse the base URL to get its domain
    parsed_base = urlparse.urlparse(app_base_url)
    base_domain = parsed_base.netloc
    
    # Remove www. prefix if present for comparison
    base_domain_clean = base_domain[4:] if base_domain.startswith('www.') else base_domain
    
    # Check if redirect_param is an absolute URL
    if redirect_param.startswith(('http://', 'https://')):
        # Parse the redirect URL
        parsed_redirect = urlparse.urlparse(redirect_param)
        redirect_domain = parsed_redirect.netloc
        
        # Remove www. prefix if present for comparison
        redirect_domain_clean = redirect_domain[4:] if redirect_domain.startswith('www.') else redirect_domain
        
        # Check if it's the same domain as base or in allowed domains
        if redirect_domain_clean != base_domain_clean and redirect_domain_clean not in ALLOWED_DOMAINS:
            raise ValueError(f"Redirect domain '{redirect_domain}' is not allowed")
        
        return redirect_param
    else:
        # It's a relative path, construct the full URL
        if not redirect_param.startswith('/'):
            redirect_param = '/' + redirect_param
        
        return urlparse.urljoin(app_base_url, redirect_param)
