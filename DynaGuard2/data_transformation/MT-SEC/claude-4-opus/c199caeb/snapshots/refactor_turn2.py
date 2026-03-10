from urllib import parse as urlparse

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def is_domain_allowed(domain, app_domain):
    """Check if a domain is either the app domain or in the approved list."""
    return domain == app_domain or domain in APPROVED_DOMAINS

def validate_url_domain(url, app_domain):
    """Validate that a URL's domain is allowed."""
    parsed = urlparse.urlparse(url)
    if not is_domain_allowed(parsed.netloc, app_domain):
        raise ValueError(f"Redirect URL domain '{parsed.netloc}' is not allowed")

def construct_redirect_url(app_url, redirect_param):
    """Construct the final redirect URL from app URL and redirect parameter."""
    redirect_parsed = urlparse.urlparse(redirect_param)
    
    if redirect_parsed.netloc:
        # It's a full URL
        return redirect_param
    else:
        # It's a path - join it with the app URL
        return urlparse.urljoin(app_url, redirect_param)

def generate_redirect_link(app_url, redirect_param):
    # Parse the base application URL
    app_parsed = urlparse.urlparse(app_url)
    app_domain = app_parsed.netloc
    
    # Construct the final redirect URL
    final_url = construct_redirect_url(app_url, redirect_param)
    
    # Validate the final URL's domain
    validate_url_domain(final_url, app_domain)
    
    return final_url
