import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def build_oauth_redirect_url(main_url, redirect_path, nonce):
    # Ensure the main_url starts with https://
    if not main_url.startswith("https://"):
        raise ValueError("main_url must start with https:// for secure connections")
    
    # Ensure nonce is provided
    if not nonce:
        raise ValueError("nonce parameter is required for OAuth integrity")
    
    # Combine the main URL with the redirect path
    full_url = urllib.parse.urljoin(main_url, redirect_path)
    
    # Parse the URL to get the domain
    parsed_url = urllib.parse.urlparse(full_url)
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Domain {domain} is not in the list of allowed callback domains")
    
    # Add the nonce as a state query parameter
    query_params = {'state': nonce}
    query_string = urllib.parse.urlencode(query_params)
    
    # Construct the final URL
    if '?' in full_url:
        final_url = f"{full_url}&{query_string}"
    else:
        final_url = f"{full_url}?{query_string}"
    
    return final_url
