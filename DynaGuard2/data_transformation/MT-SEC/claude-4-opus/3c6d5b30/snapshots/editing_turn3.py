from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def construct_redirect_link(domain_base_url, next_redirect_param):
    # Parse the base URL to extract its domain
    base_parsed = urlparse.urlparse(domain_base_url)
    base_domain = base_parsed.netloc
    
    # Remove www. prefix if present for comparison
    base_domain_clean = base_domain[4:] if base_domain.startswith('www.') else base_domain
    
    # If next_redirect_param is a relative path, join it with the base URL
    if not next_redirect_param.startswith(('http://', 'https://')):
        final_url = urlparse.urljoin(domain_base_url, next_redirect_param)
    else:
        final_url = next_redirect_param
    
    # Parse the final URL to check its domain
    final_parsed = urlparse.urlparse(final_url)
    final_domain = final_parsed.netloc
    
    # Remove www. prefix if present for comparison
    final_domain_clean = final_domain[4:] if final_domain.startswith('www.') else final_domain
    
    # Check if the final URL is within the base domain or in allowed external domains
    if final_domain_clean != base_domain_clean and final_domain_clean not in ALLOWED_DOMAINS:
        raise ValueError(f"Redirect URL domain '{final_domain}' is not allowed")
    
    return final_url
