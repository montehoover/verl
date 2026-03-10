from urllib.parse import urljoin, urlencode

def build_oauth_redirect_url(main_url, redirect_path, nonce):
    # Combine the main URL with the redirect path
    full_url = urljoin(main_url, redirect_path)
    
    # Add the nonce as a state query parameter
    query_params = {'state': nonce}
    query_string = urlencode(query_params)
    
    # Construct the final URL
    if '?' in full_url:
        final_url = f"{full_url}&{query_string}"
    else:
        final_url = f"{full_url}?{query_string}"
    
    return final_url
