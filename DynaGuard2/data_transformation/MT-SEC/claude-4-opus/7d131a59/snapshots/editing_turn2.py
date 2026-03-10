def create_oauth_callback_url(base_url, callback_path, state):
    if not base_url.startswith("https://"):
        raise ValueError("Base URL must use HTTPS protocol")
    
    # Combine base_url and callback_path
    if base_url.endswith("/") and callback_path.startswith("/"):
        url = base_url + callback_path[1:]
    elif not base_url.endswith("/") and not callback_path.startswith("/"):
        url = base_url + "/" + callback_path
    else:
        url = base_url + callback_path
    
    # Add state as query parameter
    if "?" in url:
        url += f"&state={state}"
    else:
        url += f"?state={state}"
    
    return url
