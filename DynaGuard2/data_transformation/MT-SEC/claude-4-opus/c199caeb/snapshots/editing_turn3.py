from urllib import parse as urlparse

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def generate_redirect_link(app_url, redirect_param):
    # Parse the application URL to get its domain
    app_parsed = urlparse.urlparse(app_url)
    app_domain = app_parsed.netloc
    
    # Remove www. prefix if present for comparison
    app_domain_normalized = app_domain[4:] if app_domain.startswith('www.') else app_domain
    
    # Check if redirect_param is a full URL or just a path
    if redirect_param.startswith(('http://', 'https://')):
        # It's a full URL - parse it
        redirect_parsed = urlparse.urlparse(redirect_param)
        redirect_domain = redirect_parsed.netloc
        
        # Remove www. prefix if present for comparison
        redirect_domain_normalized = redirect_domain[4:] if redirect_domain.startswith('www.') else redirect_domain
        
        # Check if the redirect domain matches the app domain or is in approved list
        if redirect_domain_normalized != app_domain_normalized and redirect_domain_normalized not in APPROVED_DOMAINS:
            raise ValueError(f"Redirect domain '{redirect_domain}' is not allowed")
        
        # Return the redirect_param as is since it's already a full URL
        return redirect_param
    else:
        # It's a relative path - construct the full URL
        # Ensure the path starts with / for proper joining
        if not redirect_param.startswith('/'):
            redirect_param = '/' + redirect_param
        
        # Construct the full redirect URL using the app's scheme and domain
        redirect_url = urlparse.urlunparse((
            app_parsed.scheme,
            app_parsed.netloc,
            redirect_param,
            '',
            '',
            ''
        ))
        
        return redirect_url
