from urllib import parse as urlparse

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

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
        # Check if it's the same domain as main_url or in permitted domains
        if target_parsed.netloc != main_domain and target_parsed.netloc not in PERMITTED_DOMAINS:
            raise ValueError(f"Redirect URL domain '{target_parsed.netloc}' is not allowed")
        # Return the full URL as is
        return target_param
    
    # If target_param is a relative path
    # Ensure it starts with / for proper joining
    if not target_param.startswith('/'):
        target_param = '/' + target_param
    
    # Construct the full URL
    result = urlparse.urljoin(main_url, target_param)
    
    # Verify the final URL stays within allowed domains
    final_parsed = urlparse.urlparse(result)
    if final_parsed.netloc != main_domain and final_parsed.netloc not in PERMITTED_DOMAINS:
        raise ValueError(f"Redirect URL domain '{final_parsed.netloc}' is not allowed")
    
    return result
