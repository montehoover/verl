import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def build_sso_url(sso_path, uid, redirect_url):
    """
    Constructs an SSO redirect URL that facilitates authentication with an external identity provider.
    
    Args:
        sso_path: a string indicating the SSO endpoint URL.
        uid: a string representing the unique user ID.
        redirect_url: a string pointing to the return URL after SSO completes.
    
    Returns:
        A string containing the full SSO redirect URL.
    
    Raises:
        ValueError: raised if the redirect_url's domain is not part of the approved domains list.
    """
    # Parse the redirect URL to extract the domain
    parsed_redirect = urllib.parse.urlparse(redirect_url)
    redirect_domain = parsed_redirect.netloc
    
    # Check if the domain is in the allowed list
    if redirect_domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Domain '{redirect_domain}' is not in the list of allowed return domains")
    
    # Construct the query parameters
    params = {
        'user': uid,
        'return': redirect_url
    }
    
    # Encode the parameters and append to the SSO path
    query_string = urllib.parse.urlencode(params)
    
    # Combine the SSO path with the query string
    if '?' in sso_path:
        # If SSO path already has query parameters, append with &
        full_url = f"{sso_path}&{query_string}"
    else:
        # Otherwise, append with ?
        full_url = f"{sso_path}?{query_string}"
    
    return full_url
