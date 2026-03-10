def assemble_oauth_callback_url(root_url, path_for_callback):
    # Remove trailing slash from root_url if present
    if root_url.endswith('/'):
        root_url = root_url[:-1]
    
    # Ensure path_for_callback starts with /
    if not path_for_callback.startswith('/'):
        path_for_callback = '/' + path_for_callback
    
    return root_url + path_for_callback
