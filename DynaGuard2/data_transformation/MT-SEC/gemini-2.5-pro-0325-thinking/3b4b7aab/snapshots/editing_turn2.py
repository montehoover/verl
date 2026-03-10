from typing import Dict

def build_url_with_query(base_url: str, path: str, query_params: Dict[str, str]) -> None:
    """
    Logs the base URL, path, and query parameters to a file named 'url_log.txt'.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path segment (e.g., "api/users").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "name": "user"}).
    """
    log_entry = f"Base URL: {base_url}, Path: {path}, Query Params: {query_params}\n"
    
    try:
        with open("url_log.txt", "a") as log_file:
            log_file.write(log_entry)
    except IOError as e:
        print(f"Error writing to log file: {e}")
