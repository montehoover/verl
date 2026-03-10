def log_url_components(base_url: str, path: str, query_params: dict) -> None:
    """
    Logs the base URL, path, and query parameters to a file named 'url_log.txt'.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component of the URL (e.g., "/api/users").
        query_params: A dictionary of query parameters (e.g., {"id": 123, "name": "test"}).
    """
    log_entry = f"Base URL: {base_url}\nPath: {path}\nQuery Params: {query_params}\n---\n"
    
    with open("url_log.txt", "a") as log_file:
        log_file.write(log_entry)
