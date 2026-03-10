def build_url_with_query(base_url, path, query_params):
    # Open the log file in append mode
    with open('url_log.txt', 'a') as log_file:
        # Write log entry
        log_file.write(f"Base URL: {base_url}\n")
        log_file.write(f"Path: {path}\n")
        log_file.write(f"Query Parameters: {query_params}\n")
        log_file.write("-" * 50 + "\n")
