from urllib.parse import urlencode, urljoin

def build_url_with_query(base_url, path, query_params):
    # Open file in append mode to add log entries
    with open('url_log.txt', 'a') as f:
        # Write log entry
        f.write(f"Base URL: {base_url}\n")
        f.write(f"Path: {path}\n")
        f.write(f"Query Parameters: {query_params}\n")
        f.write("-" * 50 + "\n")
