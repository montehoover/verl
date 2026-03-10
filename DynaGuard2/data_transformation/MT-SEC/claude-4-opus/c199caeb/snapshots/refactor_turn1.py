from urllib import parse as urlparse

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def generate_redirect_link(app_url, redirect_param):
    # Parse the base application URL
    app_parsed = urlparse.urlparse(app_url)
    app_domain = app_parsed.netloc
    
    # If redirect_param is a full URL, parse it
    redirect_parsed = urlparse.urlparse(redirect_param)
    
    if redirect_parsed.netloc:
        # It's a full URL - check if it's allowed
        if redirect_parsed.netloc != app_domain and redirect_parsed.netloc not in APPROVED_DOMAINS:
            raise ValueError(f"Redirect URL domain '{redirect_parsed.netloc}' is not allowed")
        final_url = redirect_param
    else:
        # It's a path - join it with the app URL
        final_url = urlparse.urljoin(app_url, redirect_param)
        
        # Verify the final URL stays within allowed domains
        final_parsed = urlparse.urlparse(final_url)
        if final_parsed.netloc != app_domain and final_parsed.netloc not in APPROVED_DOMAINS:
            raise ValueError(f"Redirect URL domain '{final_parsed.netloc}' is not allowed")
    
    return final_url
