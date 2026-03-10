from urllib import parse as urlparse

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def construct_redirect_url(main_url, target_param):
    # Parse the main URL to get the application's domain
    main_parsed = urlparse.urlparse(main_url)
    main_domain = main_parsed.netloc
    
    # Handle www prefix for main domain
    if main_domain.startswith('www.'):
        main_domain = main_domain[4:]
    
    # Parse the target parameter
    if target_param.startswith('http://') or target_param.startswith('https://'):
        # It's a full URL
        target_parsed = urlparse.urlparse(target_param)
        target_domain = target_parsed.netloc
        
        # Handle www prefix for target domain
        if target_domain.startswith('www.'):
            target_domain = target_domain[4:]
        
        # Check if target domain is allowed
        if target_domain != main_domain and target_domain not in PERMITTED_DOMAINS:
            raise ValueError(f"Target domain '{target_domain}' is not permitted for redirection")
        
        # Return the full URL as-is
        return target_param
    else:
        # It's a path, so combine with main URL
        if not target_param.startswith('/'):
            target_param = '/' + target_param
        
        # Construct the final URL
        final_url = urlparse.urlunparse((
            main_parsed.scheme,
            main_parsed.netloc,
            target_param,
            '',
            '',
            ''
        ))
        
        return final_url
